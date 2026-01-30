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

# Biblioteki do wykresów
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class AnalyticsEngine:
    def __init__(self):
        self.data = []
        self.start_t = 0

    def reset_timer(self):
        self.start_t = time.perf_counter()

    def get_elapsed(self):
        return (time.perf_counter() - self.start_t) * 1000  # Wynik w ms

    def log_frame(self, frame_idx, res, is_ov, steps, fps, cpu, mem):
        self.data.append({
            'frame': frame_idx,
            'res': res,
            'openvino': is_ov,
            'steps': steps, 
            'fps': fps,
            'cpu': cpu,
            'mem': mem,
            'device': 'GPU' if torch.cuda.is_available() else 'CPU'
        })

    def save_report(self):
        import csv
        if not self.data: return
        with open("benchmark_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)

# Tracker
class LowLevelTracker:
    def __init__(self, max_lost=30, iou_thresh=0.3):
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.tracks = {} 
        self.next_id = 1
        self.pos_hist = defaultdict(lambda: deque(maxlen=30))

    def get_feats(self, box, shape):
        x1, y1, x2, y2 = box
        w, h = shape
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = x2-x1, y2-y1
        return np.array([cx/w, cy/h, bw/w, bh/h, (bw/bh if bh else 1.0)], dtype=np.float32)

    def calc_iou(self, boxesA, boxesB):
        boxesA = np.array(boxesA, dtype=np.float32)
        boxesB = np.array(boxesB, dtype=np.float32)
        if len(boxesA) == 0 or len(boxesB) == 0:
            return np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)
        xA = np.maximum(boxesA[:, None, 0], boxesB[None, :, 0])
        yA = np.maximum(boxesA[:, None, 1], boxesB[None, :, 1])
        xB = np.minimum(boxesA[:, None, 2], boxesB[None, :, 2])
        yB = np.minimum(boxesA[:, None, 3], boxesB[None, :, 3])
        inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        areaA = (boxesA[:, 2] - boxesA[:, 0]) * (boxesA[:, 3] - boxesA[:, 1])
        areaB = (boxesB[:, 2] - boxesB[:, 0]) * (boxesB[:, 3] - boxesB[:, 1])
        union = areaA[:, None] + areaB[None, :] - inter
        return inter / (union + 1e-6)

    def predict_pos(self):
        preds = []
        ids = []
        for tid, info in self.tracks.items():
            if info['status'] == 'lost': continue
            box = info['box']
            hist = self.pos_hist[tid]
            dx, dy = 0, 0
            if len(hist) >= 2:
                dx = hist[-1][0] - hist[-2][0]
                dy = hist[-1][1] - hist[-2][1]
            preds.append((box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy))
            ids.append(tid)
        return preds, ids

    def get_interpolated_tracks(self):
        current_state = {}
        for tid, info in self.tracks.items():
            if info['status'] == 'active':
                box = info['box']
                hist = self.pos_hist[tid]
                dx, dy = 0, 0
                if len(hist) >= 2:
                    dx = hist[-1][0] - hist[-2][0]
                    dy = hist[-1][1] - hist[-2][1]
                new_box = (int(box[0]+dx), int(box[1]+dy), int(box[2]+dx), int(box[3]+dy))
                current_state[tid] = new_box
        return current_state

    def update(self, dets, img_shape):
        preds, active_ids = self.predict_pos()
        matches = []
        unmatched_dets = set(range(len(dets)))
        unmatched_tracks = set(active_ids)

        if len(dets) > 0 and len(preds) > 0:
            iou_mat = self.calc_iou(dets, preds)
            sorted_idx = np.dstack(np.unravel_index(np.argsort(-iou_mat.ravel()), iou_mat.shape))[0]
            used_d, used_t = set(), set()
            for d_idx, t_idx in sorted_idx:
                if iou_mat[d_idx, t_idx] < self.iou_thresh: break
                tid = active_ids[t_idx]
                if d_idx in used_d or tid in used_t: continue
                matches.append((d_idx, tid))
                used_d.add(d_idx)
                used_t.add(tid)
            unmatched_dets -= used_d
            unmatched_tracks -= used_t

        for d_idx, tid in matches:
            box = dets[d_idx]
            self.tracks[tid].update({
                'box': box, 
                'feats': self.get_feats(box, img_shape),
                'status': 'active', 
                'lost_cnt': 0
            })
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            self.pos_hist[tid].append((cx, cy))

        for d_idx in unmatched_dets:
            box = dets[d_idx]
            feats = self.get_feats(box, img_shape)
            best_id, best_dist = None, float('inf')
            for tid, info in self.tracks.items():
                if info['status'] == 'lost':
                    dist = np.linalg.norm(feats - info['feats'])
                    if dist < best_dist and dist < 0.2: 
                        best_dist = dist
                        best_id = tid
            if best_id: 
                tid = best_id
                self.tracks[tid].update({'box': box, 'feats': feats, 'status': 'active', 'lost_cnt': 0})
            else: 
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': box, 'feats': feats, 'status': 'active', 'lost_cnt': 0}
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            self.pos_hist[tid].append((cx, cy))

        lost_ids = []
        for tid in unmatched_tracks:
            self.tracks[tid]['status'] = 'lost'
            self.tracks[tid]['lost_cnt'] += 1

        to_del = [tid for tid, info in self.tracks.items() if info['lost_cnt'] > self.max_lost]
        for tid in to_del: 
            del self.tracks[tid]
            lost_ids.append(tid)

        active_tracks_map = {}
        track_id_to_det_index = {} 
        for d_idx, tid in matches:
            active_tracks_map[tid] = self.tracks[tid]['box']
            track_id_to_det_index[tid] = d_idx
        for d_idx in unmatched_dets:
            tgt_box = dets[d_idx]
            for tid, info in self.tracks.items():
                if np.array_equal(info['box'], tgt_box):
                    active_tracks_map[tid] = tgt_box
                    track_id_to_det_index[tid] = d_idx
                    break
        return active_tracks_map, lost_ids, track_id_to_det_index

def numpy_erode(mask, iterations=1):
    if iterations <= 0: return mask
    
    m = mask.astype(bool)
    
    for _ in range(iterations):
        m[1:-1, 1:-1] &= m[0:-2, 1:-1] # Góra
        m[1:-1, 1:-1] &= m[2:, 1:-1]   # Dół
        m[1:-1, 1:-1] &= m[1:-1, 0:-2] # Lewo
        m[1:-1, 1:-1] &= m[1:-1, 2:]   # Prawo

        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
        
    return m.astype(np.uint8)

def get_color_histogram_method(img, mask, box):
    x1, y1, x2, y2 = map(int, box) 
    h, w = img.shape[:2]
    
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    roi_img = img[y1:y2, x1:x2]
    roi_h, roi_w = roi_img.shape[:2]
    if roi_h < 5 or roi_w < 5: return None, None
    
    mask_resized = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0.5).astype(np.uint8)

    mask_area = np.sum(mask_bin)
    iterations = max(1, int(mask_area / 5000)) if roi_w > 40 else 0
    mask_eroded = numpy_erode(mask_bin, iterations=iterations)

    center_y, center_x = int(roi_h * 0.35), int(roi_w * 0.5)
    crop_h, crop_w = int(roi_h * 0.25), int(roi_w * 0.60) 
    
    y_start, y_end = max(0, center_y - crop_h // 2), min(roi_h, center_y + crop_h // 2)
    x_start, x_end = max(0, center_x - crop_w // 2), min(roi_w, center_x + crop_w // 2)

    img_crop = roi_img[y_start:y_end, x_start:x_end]
    mask_crop = mask_eroded[y_start:y_end, x_start:x_end]

    valid_pixels = img_crop[mask_crop > 0]
    
    if len(valid_pixels) < 10:
        mask_crop = mask_bin[y_start:y_end, x_start:x_end]
        valid_pixels = img_crop[mask_crop > 0]
        
    if len(valid_pixels) < 10: return "Unknown", None 

    valid_pixels_2d = valid_pixels.reshape(-1, 1, 3)
    hsv_pixels = cv2.cvtColor(valid_pixels_2d, cv2.COLOR_BGR2HSV)
    
    H, S, V = hsv_pixels[:,:,0].flatten(), hsv_pixels[:,:,1].flatten(), hsv_pixels[:,:,2].flatten()
    scores = defaultdict(int)
    
    is_black = (V < 35) 
    scores['Black'] = np.sum(is_black)
    is_white = (V > 200) & (S < 40)
    scores['White'] = np.sum(is_white)
    is_gray = (S < 30) & (~is_white) & (~is_black)
    scores['Gray'] = np.sum(is_gray)

    is_color = (~is_black) & (~is_white) & (~is_gray)
    H_color = H[is_color]
    
    if len(H_color) > 0:
        scores['Red'] = np.sum((H_color < 10) | (H_color > 170))
        scores['Yellow'] = np.sum((H_color >= 10) & (H_color < 35))
        scores['Green'] = np.sum((H_color >= 35) & (H_color < 85))
        scores['Blue'] = np.sum((H_color >= 85) & (H_color < 135))
        scores['Purple'] = np.sum((H_color >= 135) & (H_color < 170))

    best_color = max(scores, key=scores.get)
    if scores[best_color] < len(H) * 0.15: return "Unknown", None
    
    return best_color, (x_start, y_start, x_end-x_start, y_end-y_start)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("People Tracker")
        self.root.geometry("1200x950")
        self.root.config(bg="#121212")
        self.bench = AnalyticsEngine()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#333", foreground="white")
        style.map("TButton", background=[('active', '#555')])
        style.configure("Horizontal.TScale", background="#1e1e1e", troughcolor="#333", bordercolor="#1e1e1e")
        style.configure("TRadiobutton", background="#1e1e1e", foreground="white", font=("Segoe UI", 9))

        self.debug_folder = "debug_histogram"
        if not os.path.exists(self.debug_folder): os.makedirs(self.debug_folder)
        self.saved_examples_count = 0
        self.saved_ids = set()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print(">>> GPU DETECTED")
        else:
            print(">>> CPU DETECTED")

        self.model = None 
        self.tracker = LowLevelTracker(max_lost=30) 
        self.process = psutil.Process()
        self.is_openvino = False
        
        self.passed = 0
        self.seen_ids = set()
        self.zone_rel = (0.5, 0, 0.3, 1)

        self.track_metadata = {}
        self.closed_tracks_stats = [] 
        
        self.fps_history = []
        self.time_history = []
        self.start_time_ref = 0

        self.cap = None
        self.play = False
        self.is_paused = False 
        self.path = None 
        
        self.current_imgsz = 320 
        self.current_skip = 2

        self.var_res = tk.IntVar(value=self.current_imgsz)
        self.var_skip = tk.IntVar(value=self.current_skip)
        self.var_mode = tk.StringVar(value="seg") 
        
        self.control_visible = True
        self.perf_section_visible = False 
        self.ext_stats_visible = False
        self.sys_stats_visible = True 

        self.setup_ui()
        self.reload_model_based_on_mode()


    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.vid_frame = tk.Frame(main_frame, bg="black", bd=2, relief="sunken")
        self.vid_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5)
        self.lbl_vid = tk.Label(self.vid_frame, bg="black", text="Load Video to Start", fg="gray")
        self.lbl_vid.pack(expand=True, fill="both")

        side_panel = tk.Frame(main_frame, bg="#1e1e1e", width=320)
        side_panel.grid(row=0, column=1, sticky="ns", padx=5)
        side_panel.pack_propagate(False)

        ctrl_header = tk.Frame(side_panel, bg="#1e1e1e")
        ctrl_header.pack(fill="x", padx=10, pady=(10,5))
        self.btn_toggle_ctrl = tk.Button(ctrl_header, text="▼", font=("Consolas", 10), bg="#1e1e1e", fg="white", bd=0, command=self.toggle_control_panel, cursor="hand2")
        self.btn_toggle_ctrl.pack(side="left")
        tk.Label(ctrl_header, text="Control Panel", bg="#1e1e1e", fg="#00d4ff", font=("Segoe UI", 14, "bold")).pack(side="left", padx=5)

        self.control_content_frame = tk.Frame(side_panel, bg="#1e1e1e")
        self.control_content_frame.pack(fill="x", padx=10, pady=5)

        btn_frame = tk.Frame(self.control_content_frame, bg="#1e1e1e")
        btn_frame.pack(pady=5, fill="x")
        ttk.Button(btn_frame, text="📂 Load", command=self.load).grid(row=0, column=0, padx=5, sticky="ew")
        ttk.Button(btn_frame, text="📷 Cam", command=self.use_camera).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(btn_frame, text="▶ Start", command=self.start).grid(row=1, column=0, padx=5, sticky="ew")
        self.btn_pause = ttk.Button(btn_frame, text="⏸ Pause", command=self.toggle_pause)
        self.btn_pause.grid(row=1, column=1, padx=5, sticky="ew")
        
        ttk.Button(btn_frame, text="⏹ Stop & Graph", command=self.stop).grid(row=2, column=0, padx=5, sticky="ew")
        ttk.Button(btn_frame, text="⟳ Reset", command=self.reset).grid(row=2, column=1, padx=5, sticky="ew")
        
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.btn_opt = tk.Button(self.control_content_frame, text="⚡ Optimize Current Mode", 
                                 command=self.optimize_model_b, bg="#444", fg="#00ff00", relief="flat")
        self.btn_opt.pack(pady=5, fill="x")

        mode_frame = tk.Frame(self.control_content_frame, bg="#1e1e1e")
        mode_frame.pack(fill="x", pady=10)
        tk.Label(mode_frame, text="Detection Mode:", bg="#1e1e1e", fg="white", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Standard (Fast, Box Only)", value="box", variable=self.var_mode, command=self.on_mode_change).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Segmentation (Color, Slow)", value="seg", variable=self.var_mode, command=self.on_mode_change).pack(anchor="w")

        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)
        self.stats_container = tk.Frame(side_panel, bg="#2d2d2d", padx=10, pady=10)
        self.stats_container.pack(fill="x", padx=10)
        self.lbl_count = tk.Label(self.stats_container, text="0", bg="#2d2d2d", fg="#00ff00", font=("Consolas", 24, "bold"))
        self.lbl_count.pack()
        tk.Label(self.stats_container, text="People Passed", bg="#2d2d2d", fg="#aaaaaa", font=("Segoe UI", 9)).pack()

        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)
        ext_header = tk.Frame(side_panel, bg="#1e1e1e")
        ext_header.pack(fill="x", padx=10, pady=5)
        self.btn_toggle_ext = tk.Button(ext_header, text="▶", font=("Consolas", 10), bg="#1e1e1e", fg="white", bd=0, command=self.toggle_ext_stats, cursor="hand2")
        self.btn_toggle_ext.pack(side="left")
        tk.Label(ext_header, text="Extended Analytics", bg="#1e1e1e", fg="#e0e0e0", font=("Segoe UI", 11, "bold")).pack(side="left", padx=5)

        self.ext_content_frame = tk.Frame(side_panel, bg="#252525", padx=10, pady=10)
        self.lbl_avg_time = tk.Label(self.ext_content_frame, text="Avg Time: 0.0s", bg="#252525", fg="#ffcc00", font=("Consolas", 10))
        self.lbl_avg_time.pack(anchor="w")
        self.lbl_avg_speed = tk.Label(self.ext_content_frame, text="Avg Speed: 0 px/s", bg="#252525", fg="#00ccff", font=("Consolas", 10))
        self.lbl_avg_speed.pack(anchor="w")
        tk.Label(self.ext_content_frame, text="Top Shirt Colors:", bg="#252525", fg="#aaa", font=("Segoe UI", 9)).pack(anchor="w", pady=(5,0))
        self.lbl_colors = tk.Label(self.ext_content_frame, text="-", bg="#252525", fg="white", font=("Consolas", 9), justify="left")
        self.lbl_colors.pack(anchor="w")

        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)
        perf_header_frame = tk.Frame(side_panel, bg="#1e1e1e")
        perf_header_frame.pack(fill="x", padx=10, pady=5)
        self.btn_toggle_perf = tk.Button(perf_header_frame, text="▶", font=("Consolas", 10), bg="#1e1e1e", fg="white", bd=0, command=self.toggle_perf_section, cursor="hand2")
        self.btn_toggle_perf.pack(side="left")
        tk.Label(perf_header_frame, text="Settings", bg="#1e1e1e", fg="white", font=("Segoe UI", 11, "bold")).pack(side="left", padx=5)

        self.perf_content_frame = tk.Frame(side_panel, bg="#1e1e1e")
        self.scale_res = tk.Scale(self.perf_content_frame, from_=160, to=640, resolution=32, orient="horizontal", label="Resolution (px)",
                                  variable=self.var_res, bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_params)
        self.scale_res.pack(fill="x", padx=10)
        self.scale_skip = tk.Scale(self.perf_content_frame, from_=0, to=10, orient="horizontal", 
                                   label="Skip Frames (0-10)", variable=self.var_skip, 
                                   bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_params)
        self.scale_skip.pack(fill="x", padx=10, pady=(10,0))
        
        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)
        sys_header = tk.Frame(side_panel, bg="#1e1e1e")
        sys_header.pack(fill="x", padx=10, pady=5)
        self.btn_toggle_sys = tk.Button(sys_header, text="▼", font=("Consolas", 10), bg="#1e1e1e", fg="white", bd=0, command=self.toggle_sys_stats, cursor="hand2")
        self.btn_toggle_sys.pack(side="left")
        tk.Label(sys_header, text="System Monitor", bg="#1e1e1e", fg="#e0e0e0", font=("Segoe UI", 11, "bold")).pack(side="left", padx=5)

        self.sys_content_frame = tk.Frame(side_panel, bg="#1e1e1e")
        self.sys_content_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_perf = tk.Label(self.sys_content_frame, text="Waiting...", bg="#1e1e1e", fg="#555", font=("Consolas", 9), justify="left")
        self.lbl_perf.pack(anchor="w", padx=10, pady=5)

        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def toggle_control_panel(self):
        if self.control_visible:
            self.control_content_frame.pack_forget()
            self.btn_toggle_ctrl.config(text="▶")
            self.control_visible = False
        else:
            self.control_content_frame.pack(fill="x", padx=10, pady=5, after=self.btn_toggle_ctrl.master)
            self.btn_toggle_ctrl.config(text="▼")
            self.control_visible = True

    def toggle_ext_stats(self):
        if self.ext_stats_visible:
            self.ext_content_frame.pack_forget()
            self.btn_toggle_ext.config(text="▶")
            self.ext_stats_visible = False
        else:
            self.ext_content_frame.pack(fill="x", pady=5, after=self.btn_toggle_ext.master)
            self.btn_toggle_ext.config(text="▼")
            self.ext_stats_visible = True

    def toggle_perf_section(self):
        if self.perf_section_visible:
            self.perf_content_frame.pack_forget()
            self.btn_toggle_perf.config(text="▶")
            self.perf_section_visible = False
        else:
            self.perf_content_frame.pack(fill="x", pady=5, after=self.btn_toggle_perf.master)
            self.btn_toggle_perf.config(text="▼")
            self.perf_section_visible = True

    def toggle_sys_stats(self):
        if self.sys_stats_visible:
            self.sys_content_frame.pack_forget()
            self.btn_toggle_sys.config(text="▶")
            self.sys_stats_visible = False
        else:
            self.sys_content_frame.pack(fill="x", pady=5, after=self.btn_toggle_sys.master)
            self.btn_toggle_sys.config(text="▼")
            self.sys_stats_visible = True

    def reload_model_based_on_mode(self):
        mode = self.var_mode.get()
        try:
            if mode == "box":
                self.model = YOLO("yolov8n.pt")
                self.btn_opt.config(text="⚡ Optimize (Standard)")
            else:
                self.model = YOLO("yolov8n-seg.pt")
                self.btn_opt.config(text="⚡ Optimize (Seg)")
            self.is_openvino = False 
            self.btn_opt.config(state="normal")
        except Exception as e:
            print(f"Error loading model: {e}")

    def on_mode_change(self):
        self.reload_model_based_on_mode()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        txt = "▶ Resume" if self.is_paused else "⏸ Pause"
        self.btn_pause.config(text=txt)

    def update_params(self, _=None):
        self.current_imgsz = self.var_res.get()
        self.current_skip = self.var_skip.get()

    def optimize_model_b(self):
        mode = self.var_mode.get()
        model_name = "yolov8n.pt" if mode == "box" else "yolov8n-seg.pt"
        export_folder = f"{model_name.replace('.pt', '')}_openvino_model"

        if messagebox.askyesno("Optimize", f"Re-optimize {model_name} to OpenVINO ({self.current_imgsz}px)?"):
            try:
                try: self.root.config(cursor="watch"); self.root.update()
                except: pass

                self.model = YOLO(model_name) 
                self.model.export(format="openvino", imgsz=self.current_imgsz)
                self.model = YOLO(export_folder)
                self.is_openvino = True
                
                self.btn_opt.config(text="✔ Optimized", state="disabled")
                messagebox.showinfo("Success", f"Model {model_name} optimized and loaded!")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.reload_model_based_on_mode()
            finally:
                try: self.root.config(cursor="") 
                except: pass

    def load(self):
        f = filedialog.askopenfilename()
        if f: 
            self.path = f
            self.lbl_vid.config(text=f"File Loaded:\n{f.split('/')[-1]}")

    def use_camera(self):
        self.path = 0 
        self.lbl_vid.config(text="Camera Selected\nPress Start")

    def start(self):
        if self.path is None: 
            messagebox.showwarning("No Source", "Please load a video first.")
            return
        
        self.play = False
        time.sleep(0.2)
        
        self.cap = cv2.VideoCapture(self.path)
        self.play = True
        self.is_paused = False
        
        self.fps_history = []
        self.time_history = []
        self.start_time_ref = time.time()
        
        self.btn_pause.config(text="⏸ Pause")
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self): 
        self.play = False
        self.root.after(500, self.show_fps_chart)
    
    def show_fps_chart(self):
        if not self.fps_history: return
        
        win = tk.Toplevel(self.root)
        win.title("Session Performance Summary")
        win.geometry("600x400")
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(self.time_history, self.fps_history, color='green', linewidth=1)
        ax.set_title("Session Performance (FPS over Time)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("FPS")
        ax.grid(True)
        
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            ax.axhline(y=avg_fps, color='red', linestyle='--', label=f'Avg: {avg_fps:.1f}')
            ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def reset(self):
        self.tracker = LowLevelTracker()
        self.passed = 0
        self.seen_ids.clear()
        self.track_metadata.clear()
        self.closed_tracks_stats.clear()
        self.saved_examples_count = 0
        self.saved_ids.clear()
        self.fps_history.clear()
        self.time_history.clear()
        self.update_ui_stats()

    def update_ui_stats(self):
        self.lbl_count.config(text=str(self.passed))
        valid_durations = [d['zone_duration'] for d in self.closed_tracks_stats if d['zone_duration'] > 0.1]
        valid_speeds = [s for d in self.closed_tracks_stats for s in d['zone_speeds']]
        valid_colors = [d['final_color'] for d in self.closed_tracks_stats if d.get('final_color') and d['final_color'] != "Unknown"]
        
        for tid, meta in self.track_metadata.items():
            if meta['zone_duration'] > 0.1:
                valid_durations.append(meta['zone_duration'])
                valid_speeds.extend(meta['zone_speeds'])
                if meta.get('final_color') and meta['final_color'] != "Unknown":
                    valid_colors.append(meta['final_color'])

        avg_t = np.mean(valid_durations) if valid_durations else 0.0
        avg_s = np.mean(valid_speeds) if valid_speeds else 0.0
        
        self.lbl_avg_time.config(text=f"Avg Time: {avg_t:.1f}s")
        self.lbl_avg_speed.config(text=f"Avg Speed: {avg_s:.1f} px/s")

        if valid_colors:
            counts = Counter(valid_colors)
            most_common = counts.most_common(3)
            self.lbl_colors.config(text="\n".join([f"{k}: {v}" for k,v in most_common]))
        else:
            self.lbl_colors.config(text="- No Data -")

    def update_perf_ui(self, fps_val, mem_mb):
        base_gflops = 12.0 if self.var_mode.get() == "seg" else 8.7
        scale_factor = (self.current_imgsz ** 2) / (640 ** 2)
        est_gflops = base_gflops * scale_factor
        
        self.lbl_perf.config(text=f"FPS: {fps_val:.1f}\nRAM: {mem_mb:.0f}MB\nEst.Load: {est_gflops:.1f} GFLOPS")

    def loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dur = 1.0 / fps
        frame_idx = 0

        while self.play and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.1)
                continue

            self.bench.reset_timer()
            t_frame_start = time.time()
            
            # 1. Odczyt klatki
            ret, frame = self.cap.read()
            if not ret: break
            if self.path == 0: frame = cv2.flip(frame, 1)
            t_read = self.bench.get_elapsed()
            
            # 2. Preprocessing
            self.bench.reset_timer()
            process_frame = cv2.resize(frame, (640, 360)) 
            h, w = process_frame.shape[:2]
            t_preprocess = self.bench.get_elapsed()
            
            steps = {'read': t_read, 'preprocess': t_preprocess}
            
            tracks = {}
            active_map = {}
            id_to_det_idx = {}
            lost_ids = []
            masks = None 

            should_detect = (frame_idx % (self.current_skip + 1)) == 0
            is_seg_mode = (self.var_mode.get() == "seg")

            if should_detect:
                # 3. Inferencja YOLO
                self.bench.reset_timer()
                results = self.model.predict(process_frame, verbose=False, imgsz=self.current_imgsz, device=self.device)[0]
                steps['inference'] = self.bench.get_elapsed()
                
                # 4. Logika Trackera
                self.bench.reset_timer()
                dets = [tuple(b.xyxy[0].cpu().numpy().astype(int)) for b in results.boxes if int(b.cls[0]) == 0]
                if is_seg_mode and results.masks is not None:
                    masks = results.masks.data.cpu().numpy() 
                
                active_map, lost_ids, id_to_det_idx = self.tracker.update(dets, (w, h))
                tracks = active_map 
                steps['tracking'] = self.bench.get_elapsed()
            else:
                tracks = self.tracker.get_interpolated_tracks()

            for tid in lost_ids:
                if tid in self.track_metadata:
                    meta = self.track_metadata[tid]
                    if meta['zone_duration'] > 0.1:
                        self.closed_tracks_stats.append({
                            'zone_duration': meta['zone_duration'],
                            'zone_speeds': meta['zone_speeds'],
                            'final_color': meta.get('final_color', 'Unknown')
                        })
                    del self.track_metadata[tid]

            rx, ry, rw, rh = self.zone_rel
            zx1, zx2 = int((rx)*w), int((rx+rw)*w)

            for tid, box in tracks.items():
                if tid not in self.track_metadata:
                    self.track_metadata[tid] = {'zone_duration': 0.0, 'zone_speeds': [], 'final_color': None}
                
                meta = self.track_metadata[tid]
                bx1, by1, bx2, by2 = map(int, box)
                cx = (bx1 + bx2) // 2
                inside = zx1 <= cx <= zx2

                if inside:
                    meta['zone_duration'] += frame_dur
                    hist = self.tracker.pos_hist[tid]
                    if len(hist) > 1:
                        dist = np.linalg.norm(np.array(hist[-1]) - np.array(hist[-2]))
                        meta['zone_speeds'].append(dist / frame_dur)
                    if tid not in self.seen_ids:
                        if len(hist) > 1 and not (zx1 <= hist[-2][0] <= zx2):
                            self.seen_ids.add(tid)
                            self.passed += 1

                if is_seg_mode and should_detect:
                    is_color_missing = (meta.get('final_color') in [None, "Unknown"])
                    if is_color_missing and (masks is not None) and (tid in id_to_det_idx):
                        self.bench.reset_timer()
                        det_idx = id_to_det_idx[tid]
                        if det_idx < len(masks):
                            person_mask = masks[det_idx]
                            
                            raw_color, _ = get_color_histogram_method(process_frame, person_mask, (bx1, by1, bx2, by2))
                            
                            steps['reid'] = steps.get('reid', 0) + self.bench.get_elapsed()

                            if raw_color:
                                meta['final_color'] = raw_color

                col = (0, 255, 0) if inside else (0, 0, 255)
                thick = 2 if should_detect else 1
                
                cv2.rectangle(process_frame, (bx1, by1), (bx2, by2), col, thick)
                
                label = f"ID:{tid}"
                if meta.get('final_color') and meta['final_color'] != "Unknown": 
                    label += f" [{meta['final_color']}]"
                
                cv2.putText(process_frame, label, (bx1, by1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, thick)

            cv2.rectangle(process_frame, (zx1, 0), (zx2, h), (255, 200, 0), 2)

            t_frame_end = time.time()
            fps_real = 1.0 / (t_frame_end - t_frame_start + 1e-6)
            current_duration = t_frame_end - self.start_time_ref
            
            self.fps_history.append(fps_real)
            self.time_history.append(current_duration)

            mem = self.process.memory_info().rss / 1024**2
            cpu = psutil.cpu_percent()
            self.bench.log_frame(frame_idx, self.current_imgsz, self.is_openvino, steps, fps_real, cpu, mem)

            if frame_idx % 5 == 0:
                self.root.after(0, self.update_ui_stats)
                self.root.after(0, self.update_perf_ui, fps_real, mem)

            img_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            
            def upd(itk=img_tk):
                if not self.play: return
                self.lbl_vid.config(image=itk)
                self.lbl_vid.image = itk
            self.root.after(0, upd)
            
            frame_idx += 1
            elapsed = time.time() - t_frame_start
            wait = frame_dur - elapsed
            if wait > 0: time.sleep(wait)

        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()