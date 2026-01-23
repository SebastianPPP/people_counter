import time
import numpy as np
import pandas as pd
import psutil
import torch
import cv2
import matplotlib.pyplot as plt

class AnalyticsEngine:
    def __init__(self):
        self.data_log = [] 
        self.erosion_tests = []
        self.reset_timer()

    def reset_timer(self):
        self.start_time = time.perf_counter()

    def get_elapsed(self):
        return (time.perf_counter() - self.start_time) * 1000

    def log_frame(self, frame_idx, imgsz, is_optimized, steps_dict, fps, cpu_usage, ram_mb):
        gpu_mem = 0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2

        entry = {
            'frame': frame_idx,
            'imgsz': imgsz,
            'optimized': is_optimized,
            'fps': fps,
            'cpu_util': cpu_usage,
            'ram_mb': ram_mb,
            'gpu_mem_mb': gpu_mem,
            **steps_dict
        }
        self.data_log.append(entry)

    def test_erosion_efficiency(self, mask, my_erode_func, iterations=1):
        if mask is None: return
        t0 = time.perf_counter()
        res_my = my_erode_func(mask, iterations)
        t_my = (time.perf_counter() - t0) * 1000
        
        t1 = time.perf_counter()
        kernel = np.ones((3,3), np.uint8)
        res_cv = cv2.erode(mask, kernel, iterations=iterations)
        t_cv = (time.perf_counter() - t1) * 1000
        
        iou = np.logical_and(res_my, res_cv).sum() / (np.logical_or(res_my, res_cv).sum() + 1e-6)
        self.erosion_tests.append({'t_custom': t_my, 't_cv2': t_cv, 'iou': iou})

    def save_report(self):
        """Zapisuje dane i AUTOMATYCZNIE wyświetla wykresy."""
        if not self.data_log:
            print(">>> Brak danych do raportu.")
            return

        df = pd.DataFrame(self.data_log)
        df.to_csv("benchmark_pipeline.csv", index=False)
        
        # URUCHOMIENIE AUTOMATYCZNEJ WIZUALIZACJI
        self._generate_visual_plots(df)
        print(">>> Raport zapisany i wykresy wyświetlone.")

    def _generate_visual_plots(self, df):
        # 1. WYKRES PROFILOWANIA POTOKU (Gdzie ucieka czas?)
        cols = ['read', 'preprocess', 'inference', 'tracking', 'reid']
        existing_cols = [c for c in cols if c in df.columns]
        means = df[existing_cols].mean()

        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.pie(means, labels=existing_cols, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title("Profilowanie Potoku (Średni czas zadania)")

        # 2. WYKRES STABILNOŚCI FPS I ZUŻYCIA CPU
        plt.subplot(1, 2, 2)
        plt.plot(df['frame'], df['fps'], label='FPS', color='green')
        plt.axhline(y=df['fps'].mean(), color='r', linestyle='--', label=f'Avg FPS: {df["fps"].mean():.1f}')
        plt.xlabel("Klatka")
        plt.ylabel("Klatki na sekundę")
        plt.title("Wydajność systemu w czasie")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        # 3. WYKRES OPÓŹNIEŃ (LATENCY) DLA RE-ID
        if 'reid' in df.columns:
            plt.figure(figsize=(10, 4))
            plt.bar(df['frame'], df['reid'], color='purple', alpha=0.6)
            plt.title("Opóźnienie rozpoznawania cech (Re-ID) na klatkę")
            plt.ylabel("Czas (ms)")
            plt.xlabel("Klatka")
            plt.show()