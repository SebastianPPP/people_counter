import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
from collections import defaultdict, deque, Counter
import time
import psutil
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("Agg")


def numpy_erode(mask, iterations=1):
    if iterations <= 0: return mask
    m = mask.astype(bool)
    for _ in range(iterations):
        m[1:-1, 1:-1] &= m[0:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, 0:-2] & m[1:-1, 2:]
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m.astype(np.uint8)

def get_color_histogram_method(img, mask, box):
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2-x1 < 5 or y2-y1 < 5: return "Unknown"
        roi = img[y1:y2, x1:x2]
        m_roi = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_roi > 0.5).astype(np.uint8)
        m_ero = numpy_erode(m_bin, iterations=1)
        pixels = roi[m_ero > 0]
        if len(pixels) < 10: pixels = roi[m_bin > 0]
        if len(pixels) < 10: return "Unknown"
        hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:,:,0].flatten(), hsv[:,:,1].flatten(), hsv[:,:,2].flatten()
        scores = defaultdict(int)
        scores['Black'] = np.sum(V < 35); scores['White'] = np.sum((V > 200) & (S < 40))
        is_c = (V >= 35) & ~((V > 200) & (S < 40))
        if np.sum(is_c) > 0:
            Hc = H[is_c]
            scores['Red'] = np.sum((Hc < 10) | (Hc > 170))
            scores['Blue'] = np.sum((Hc >= 85) & (Hc < 135))
        return max(scores, key=scores.get) if scores else "Unknown"
    except: return "Unknown"

class LowLevelTracker:
    def __init__(self, max_lost=30, iou_thresh=0.3):
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.tracks = {} 
        self.next_id = 1
        self.pos_hist = defaultdict(lambda: deque(maxlen=30))

    def calc_iou(self, boxesA, boxesB):
        boxesA, boxesB = np.array(boxesA), np.array(boxesB)
        if len(boxesA) == 0 or len(boxesB) == 0: return np.zeros((len(boxesA), len(boxesB)))
        xA, yA = np.maximum(boxesA[:,None,0], boxesB[None,:,0]), np.maximum(boxesA[:,None,1], boxesB[None,:,1])
        xB, yB = np.minimum(boxesA[:,None,2], boxesB[None,:,2]), np.minimum(boxesA[:,None,3], boxesB[None,:,3])
        inter = np.maximum(0, xB-xA) * np.maximum(0, yB-yA)
        areaA, areaB = (boxesA[:,2]-boxesA[:,0])*(boxesA[:,3]-boxesA[:,1]), (boxesB[:,2]-boxesB[:,0])*(boxesB[:,3]-boxesB[:,1])
        return inter / (areaA[:,None] + areaB[None,:] - inter + 1e-6)

    def update(self, dets, img_shape):
        active_ids = [tid for tid, info in self.tracks.items() if info['status'] != 'lost']
        preds = []
        for tid in active_ids:
            box, h = self.tracks[tid]['box'], self.pos_hist[tid]
            dx, dy = (h[-1][0]-h[-2][0], h[-1][1]-h[-2][1]) if len(h) >= 2 else (0,0)
            preds.append((box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy))
        matches, unmatched_dets = [], set(range(len(dets)))
        if len(dets) > 0 and len(preds) > 0:
            iou_mat = self.calc_iou(dets, preds)
            for d_idx, t_idx in np.argwhere(iou_mat > self.iou_thresh):
                if d_idx in unmatched_dets:
                    matches.append((d_idx, active_ids[t_idx]))
                    unmatched_dets.remove(d_idx)
        for d_idx, tid in matches:
            box = dets[d_idx]
            self.tracks[tid].update({'box': box, 'status': 'active', 'lost_cnt': 0})
            self.pos_hist[tid].append(((box[0]+box[2])//2, (box[1]+box[3])//2))
        for d_idx in unmatched_dets:
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = {'box': dets[d_idx], 'status': 'active', 'lost_cnt': 0}
            self.pos_hist[tid].append(((dets[d_idx][0]+dets[d_idx][2])//2, (dets[d_idx][1]+dets[d_idx][3])//2))
        return {tid: info['box'] for tid, info in self.tracks.items() if info['status'] == 'active'}

class BenchmarkEngine:
    def __init__(self, root_app):
        self.app = root_app
        init_data = lambda: {'inf': {}, 'trk': {}, 'res': {'imgsz': [], 'fps': [], 'reid': [], 'occ': [], 'gflops': []}}
        self.results = {'cpu': init_data(), 'gpu': init_data()}

    def run_suite(self, video_path):
        devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
        resolutions = [160, 320, 480, 640]
        model = YOLO("yolov8n-seg.pt")

        for dev in devices:
            key = 'gpu' if dev == 'cuda' else 'cpu'
            self.app.update_status(f"Testing {key.upper()}...")
            
            for trk in ['Custom', 'bytetrack.yaml', 'botsort.yaml']:
                cap = cv2.VideoCapture(video_path)
                inf_l, trk_l = [], []
                tracker = LowLevelTracker()
                for _ in range(25):
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.resize(frame, (640, 360))
                    t0 = time.perf_counter()
                    res = model.predict(frame, imgsz=320, device=dev, verbose=False)[0]
                    inf_l.append((time.perf_counter()-t0)*1000)
                    t1 = time.perf_counter()
                    if trk == 'Custom':
                        tracker.update([tuple(b.xyxy[0].cpu().numpy().astype(int)) for b in res.boxes], (640,360))
                    else:
                        model.track(frame, tracker=trk, imgsz=320, device=dev, verbose=False, persist=True)
                    trk_l.append((time.perf_counter()-t1)*1000)
                cap.release()
                self.results[key]['inf'][trk] = np.mean(inf_l); self.results[key]['trk'][trk] = np.mean(trk_l)

            cap = cv2.VideoCapture(video_path)
            for r in resolutions:
                frames, t_start = 0, time.time()
                reid_buffer = defaultdict(list); tracker = LowLevelTracker()
                for _ in range(20):
                    ret, frame = cap.read()
                    if not ret: break
                    frames += 1
                    res = model.predict(frame, imgsz=r, device=dev, verbose=False)[0]
                    boxes = [tuple(b.xyxy[0].cpu().numpy().astype(int)) for b in res.boxes]
                    tracks = tracker.update(boxes, frame.shape[:2])
                    if res.masks:
                        for i, (tid, b) in enumerate(tracks.items()):
                            if i < len(res.masks.data):
                                reid_buffer[tid].append(get_color_histogram_method(frame, res.masks.data[i].cpu().numpy(), b))
                
                fps = frames / (time.time() - t_start + 1e-6)
                self.results[key]['res']['imgsz'].append(r)
                self.results[key]['res']['fps'].append(fps)
                self.results[key]['res']['gflops'].append(12.0 * ((r/640)**2) * fps)
                accs = [Counter(v).most_common(1)[0][1]/len(v) for v in reid_buffer.values() if len(v)>2]
                self.results[key]['res']['reid'].append(np.mean(accs)*100 if accs else 0)
                self.results[key]['res']['occ'].append(min(100, (frames / (tracker.next_id + 1e-6)) * 10))
            cap.release()
        self.app.show_report(self.results)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Benchmark")
        self.root.geometry("1400x900")
        self.root.configure(bg="#121212")
        self.bench = BenchmarkEngine(self)
        self.video_path = None
        self.setup_ui()

    def setup_ui(self):
        l = tk.Frame(self.root, bg="#1e1e1e", width=250); l.pack(side="left", fill="y", padx=5, pady=5); l.pack_propagate(False)
        tk.Label(l, text="BENCHMARK", bg="#1e1e1e", fg="#00ff00", font=("Arial", 12, "bold")).pack(pady=20)
        tk.Button(l, text="📂 Load Video", command=self.load_video).pack(fill="x", padx=10)
        self.btn = tk.Button(l, text="🚀 RUN SUITE", command=self.run_benchmark, state="disabled", bg="#d63031", fg="white")
        self.btn.pack(fill="x", padx=10, pady=20)
        self.st = tk.Label(l, text="Ready", bg="#1e1e1e", fg="gray"); self.st.pack(side="bottom", pady=20)
        self.fig = Figure(figsize=(11, 8), dpi=100, facecolor="#121212")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root); self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)

    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path: self.btn.config(state="normal")

    def update_status(self, t): self.st.config(text=t); self.root.update()

    def run_benchmark(self):
        self.btn.config(state="disabled")
        threading.Thread(target=lambda: self.bench.run_suite(self.video_path), daemon=True).start()

    def save_individual_plot(self, plot_func, filename, title, xlabel, ylabel):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        plot_func(ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)

    def show_report(self, res):
        self.fig.clf(); axs = self.fig.subplots(2, 3)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        r_cpu = res['cpu']['res']
        r_gpu = res['gpu']['res']

        # 1. Latency 
        def plot_latency(ax):
            trks = list(res['cpu']['inf'].keys())
            x = np.arange(len(trks))
            inf_v = [res['cpu']['inf'].get(t,0) for t in trks]
            trk_v = [res['cpu']['trk'].get(t,0) for t in trks]
            ax.bar(x, inf_v, 0.4, label='Inference (AI)', color='#ff7675')
            ax.bar(x, trk_v, 0.4, bottom=inf_v, label='Tracking (Math)', color='#74b9ff')
            ax.set_xticks(x); ax.set_xticklabels(trks, rotation=15)
            ax.legend()
        
        self.save_individual_plot(plot_latency, "report_latency.png", "Latency Breakdown", "Tracker Type", "Time (ms)")
        plot_latency(axs[0,0]); axs[0,0].set_title("1. Latency Breakdown", color='white')

        # 2. FPS Scaling
        def plot_fps(ax):
            ax.plot(r_cpu['imgsz'], r_cpu['fps'], 'o-', label='CPU FPS', color='#ff7675')
            if r_gpu['fps']:
                ax.plot(r_gpu['imgsz'], r_gpu['fps'], 's--', label='GPU FPS', color='#00cec9')
            ax.legend()

        self.save_individual_plot(plot_fps, "report_fps.png", "FPS vs Resolution", "Image Size (px)", "Frames Per Second")
        plot_fps(axs[0,1]); axs[0,1].set_title("2. FPS Scaling", color='white')

        # 3. Re-ID Accuracy
        def plot_reid(ax):
            ax.plot(r_cpu['imgsz'], r_cpu['reid'], 'o-', label='CPU Re-ID', color='#fdcb6e')
            if r_gpu['reid']:
                ax.plot(r_gpu['imgsz'], r_gpu['reid'], 's--', label='GPU Re-ID', color='#e17055')
            ax.legend()

        self.save_individual_plot(plot_reid, "report_reid.png", "Re-ID Accuracy", "Image Size (px)", "Accuracy (%)")
        plot_reid(axs[0,2]); axs[0,2].set_title("3. Re-ID Accuracy", color='white')

        # 4. ID Stability
        def plot_occ(ax):
            ax.bar([str(i) for i in r_cpu['imgsz']], r_cpu['occ'], color='#a29bfe', alpha=0.7)
        
        self.save_individual_plot(plot_occ, "report_stability.png", "ID Stability Score", "Image Size (px)", "Stability Score")
        plot_occ(axs[1,0]); axs[1,0].set_title("4. ID Stability", color='white')

        # 5. Sweet Spot
        def plot_sweet(ax):
            q, s = np.array(r_cpu['reid']), np.array(r_cpu['fps'])
            score = (q * s) / 100
            ax.plot([str(i) for i in r_cpu['imgsz']], score, 'D-g', label='Perf Score')
            ax.legend()

        self.save_individual_plot(plot_sweet, "report_sweetspot.png", "Quality/Speed Trade-off", "Image Size (px)", "Efficiency Score")
        plot_sweet(axs[1,1]); axs[1,1].set_title("5. Sweet Spot", color='white')

        # 6. GFLOPS
        def plot_gflops(ax):
            if r_gpu['gflops']:
                ax.bar([str(i) for i in r_gpu['imgsz']], r_gpu['gflops'], color='#00b894')
            else:
                ax.text(0.5, 0.5, "GPU N/A", ha='center')

        self.save_individual_plot(plot_gflops, "report_gflops.png", "GPU GFLOPS Load", "Image Size (px)", "GFLOPS")
        plot_gflops(axs[1,2]); axs[1,2].set_title("6. GPU GFLOPS", color='white')

        for a in axs.flat: 
            a.set_facecolor('#1e1e1e')
            a.tick_params(colors='white')
            a.xaxis.label.set_color('white')
            a.yaxis.label.set_color('white')

        self.canvas.draw(); self.btn.config(state="normal"); self.update_status("Saved & Displayed.")

if __name__ == "__main__":
    root = tk.Tk(); app = App(root); root.mainloop()