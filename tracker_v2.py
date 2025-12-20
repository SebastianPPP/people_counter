import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
from collections import defaultdict, deque
import time
import psutil
import torch

class LowLevelTracker:
    def __init__(self, max_lost=30, iou_thresh=0.3):
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.tracks = {} 
        self.next_id = 1
        self.pos_hist = defaultdict(lambda: deque(maxlen=5))

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
            self.tracks[tid].update({'box': box, 'feats': self.get_feats(box, img_shape), 'status': 'active', 'lost_cnt': 0})
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

        for tid in unmatched_tracks:
            self.tracks[tid]['status'] = 'lost'
            self.tracks[tid]['lost_cnt'] += 1

        to_del = [tid for tid, info in self.tracks.items() if info['lost_cnt'] > self.max_lost]
        for tid in to_del: del self.tracks[tid]

        return {tid: i['box'] for tid, i in self.tracks.items() if i['status'] == 'active'}


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent People Counter")
        self.root.geometry("1100x750")
        self.root.config(bg="#121212")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#333", foreground="white")
        style.map("TButton", background=[('active', '#555')])
        style.configure("Horizontal.TScale", background="#1e1e1e", troughcolor="#333", bordercolor="#1e1e1e")

        self.device = 'cpu'
        self.using_gpu = False
        self.current_imgsz = 320
        self.current_skip = 2 

        if torch.cuda.is_available():
            self.device = 0
            self.using_gpu = True
            print("DETECTED GPU: Enabling GPU Mode")
            self.current_imgsz = 640
            self.current_skip = 0
            self.hw_status = "GPU (CUDA)"
            self.hw_color = "#00ff00"
        else:
            print("NO GPU DETECTED: Using CPU Mode")
            self.hw_status = "CPU"
            self.hw_color = "#ffcc00"
            
        self.model = YOLO("yolov8n.pt")
        self.tracker = LowLevelTracker(max_lost=30) 
        self.process = psutil.Process()
        self.is_openvino = False
        
        self.passed = 0
        self.seen_ids = set()
        self.frames_in_zone = defaultdict(int) 
        self.speeds = [] 
        self.zone_rel = (0.5, 0, 0.3, 1)

        self.cap = None
        self.play = False
        self.path = None 
        
        self.var_res = tk.IntVar(value=self.current_imgsz)
        self.var_skip = tk.IntVar(value=self.current_skip)
        
        self.perf_section_visible = False 

        self.setup_ui()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.vid_frame = tk.Frame(main_frame, bg="black", bd=2, relief="sunken")
        self.vid_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5)
        self.lbl_vid = tk.Label(self.vid_frame, bg="black", text="No Video/Camera Loaded", fg="gray")
        self.lbl_vid.pack(expand=True, fill="both")

        side_panel = tk.Frame(main_frame, bg="#1e1e1e", width=300)
        side_panel.grid(row=0, column=1, sticky="ns", padx=5)
        side_panel.pack_propagate(False)

        tk.Label(side_panel, text="Control Panel", bg="#1e1e1e", fg="#00d4ff", font=("Segoe UI", 14, "bold")).pack(pady=15)

        btn_frame = tk.Frame(side_panel, bg="#1e1e1e")
        btn_frame.pack(pady=5, fill="x")
        ttk.Button(btn_frame, text="📂 Load File", command=self.load).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(btn_frame, text="📷 Camera", command=self.use_camera).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(btn_frame, text="▶ Start", command=self.start).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(btn_frame, text="⏹ Stop", command=self.stop).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(btn_frame, text="⟳ Reset Stats", command=self.reset).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        if not self.using_gpu:
            self.btn_opt = tk.Button(side_panel, text="⚡ Optimize for CPU (OpenVINO)", 
                                     command=self.optimize_model_b, bg="#444", fg="#00ff00", relief="flat")
            self.btn_opt.pack(pady=5, fill="x", padx=10)

        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(side_panel, text="Analytics", bg="#1e1e1e", fg="white", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10)
        self.stats_container = tk.Frame(side_panel, bg="#2d2d2d", padx=10, pady=10)
        self.stats_container.pack(fill="x", padx=10, pady=5)
        self.lbl_count = tk.Label(self.stats_container, text="0", bg="#2d2d2d", fg="#00ff00", font=("Consolas", 24, "bold"))
        self.lbl_count.pack()
        tk.Label(self.stats_container, text="People Passed", bg="#2d2d2d", fg="#aaaaaa", font=("Segoe UI", 9)).pack()

        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)

        perf_header_frame = tk.Frame(side_panel, bg="#1e1e1e")
        perf_header_frame.pack(fill="x", padx=10, pady=5)
        
        self.btn_toggle_perf = tk.Button(perf_header_frame, text="▶", font=("Consolas", 10), 
                                         bg="#1e1e1e", fg="white", bd=0, command=self.toggle_perf_section, cursor="hand2")
        self.btn_toggle_perf.pack(side="left")
        
        tk.Label(perf_header_frame, text="Performance Settings", bg="#1e1e1e", fg="white", font=("Segoe UI", 11, "bold")).pack(side="left", padx=5)

        self.perf_content_frame = tk.Frame(side_panel, bg="#1e1e1e")
        
        tk.Label(self.perf_content_frame, text="Resolution (px):", bg="#1e1e1e", fg="#ccc", font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        self.scale_res = tk.Scale(self.perf_content_frame, from_=160, to=640, resolution=32, orient="horizontal", 
                                  variable=self.var_res, bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_params)
        self.scale_res.pack(fill="x", padx=10)

        tk.Label(self.perf_content_frame, text="Skip Frames:", bg="#1e1e1e", fg="#ccc", font=("Segoe UI", 9)).pack(anchor="w", padx=10, pady=(10,0))
        self.scale_skip = tk.Scale(self.perf_content_frame, from_=0, to=10, orient="horizontal", 
                                   variable=self.var_skip, bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_params)
        self.scale_skip.pack(fill="x", padx=10)
        tk.Label(self.perf_content_frame, text="(0=Analyze all, 5=Analyze 1 of 6)", bg="#1e1e1e", fg="#666", font=("Segoe UI", 8)).pack(anchor="w", padx=10)
        
        ttk.Separator(side_panel, orient='horizontal').pack(fill='x', pady=10)

        tk.Label(side_panel, text="System Status", bg="#1e1e1e", fg="white", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=10)
        self.lbl_mode = tk.Label(side_panel, text=f"Mode: {self.hw_status}", bg="#1e1e1e", fg=self.hw_color, font=("Consolas", 10, "bold"))
        self.lbl_mode.pack(padx=10, pady=(5,0), anchor="w")
        self.lbl_perf = tk.Label(side_panel, text="Waiting...", bg="#1e1e1e", fg="#aaaaaa", font=("Consolas", 9), justify="left")
        self.lbl_perf.pack(padx=10, pady=10, anchor="w")

        main_frame.columnconfigure(0, weight=4)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def toggle_perf_section(self):
        if self.perf_section_visible:
            self.perf_content_frame.pack_forget()
            self.btn_toggle_perf.config(text="▶")
            self.perf_section_visible = False
        else:
            self.perf_content_frame.pack(fill="x", pady=5, after=self.btn_toggle_perf.master)
            self.btn_toggle_perf.config(text="▼")
            self.perf_section_visible = True

    def update_params(self, _=None):
        self.current_imgsz = self.var_res.get()
        self.current_skip = self.var_skip.get()
        if self.is_openvino:
            self.lbl_mode.config(text="Mode: OpenVINO", fg="#00ffff")
        
    def optimize_model_b(self):
        msg = f"Export to OpenVINO with resolution {self.current_imgsz}px?\n\nTip: Lower resolution = Higher FPS."
        if messagebox.askyesno("Optimize Model", msg):
            try:
                self.lbl_mode.config(text="Exporting... Wait...")
                self.root.update()
                export_sz = self.current_imgsz
                self.model = YOLO("yolov8n.pt") 
                self.model.export(format="openvino", imgsz=export_sz)
                self.model = YOLO("yolov8n_openvino_model/")
                self.is_openvino = True
                self.lbl_mode.config(text=f"Mode: CPU (OpenVINO {export_sz}px)", fg="#00ffff")
                messagebox.showinfo("Success", f"Model optimized for {export_sz}px! Restart video.")
                self.btn_opt.config(text="⚡ Re-Optimize with new settings")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}\nTry: pip install openvino")
                self.lbl_mode.config(text=f"Mode: {self.hw_status}")

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
            messagebox.showwarning("No Source", "Please load a video or select camera first.")
            return
        self.cap = cv2.VideoCapture(self.path)
        self.play = True
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self): self.play = False
    
    def reset(self):
        self.tracker = LowLevelTracker()
        self.passed = 0
        self.seen_ids.clear()
        self.frames_in_zone.clear()
        self.speeds.clear()
        self.update_stats_ui()

    def update_stats_ui(self):
        self.lbl_count.config(text=str(self.passed))

    def update_perf_ui(self, proc_ms, mem_mb):
        curr_fps = 1000 / (proc_ms + 1e-6)
        color = "#00ff00" if curr_fps >= 24 else "#ff5555"
        txt = (f"FPS: {curr_fps:.1f}\nLat: {proc_ms:.1f} ms\nRAM: {mem_mb:.0f} MB")
        self.lbl_perf.config(text=txt, fg=color)

    def loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0: fps = 30.0
        frame_dur = 1.0 / fps # czas trwania jednej klatki w sekundach

        frame_idx = 0
        while self.play and self.cap.isOpened():
            t_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret: break
            if self.path == 0: frame = cv2.flip(frame, 1)

            process_frame = cv2.resize(frame, (640, 360)) 
            h, w = process_frame.shape[:2]
            tracks = {}
            interval = self.current_skip + 1
            should_detect = (frame_idx % interval) == 0

            if should_detect:
                infer_sz = self.current_imgsz
                res = self.model.predict(process_frame, verbose=False, imgsz=infer_sz, device=self.device)[0]
                dets = [tuple(b.xyxy[0].cpu().numpy().astype(int)) for b in res.boxes if int(b.cls[0]) == 0] 
                tracks = self.tracker.update(dets, (w, h))
            else:
                tracks = self.tracker.get_interpolated_tracks()

            rx, ry, rw, rh = self.zone_rel
            zx1, zx2 = int((rx)*w), int((rx+rw)*w)
            cv2.rectangle(process_frame, (zx1, 0), (zx2, h), (255, 200, 0), 2)

            for tid, box in tracks.items():
                cx = (box[0] + box[2]) // 2
                inside = zx1 <= cx <= zx2
                if inside:
                    self.frames_in_zone[tid] += 1
                    if tid not in self.seen_ids:
                        hist = self.tracker.pos_hist[tid]
                        if len(hist) > 1 and not (zx1 <= hist[-2][0] <= zx2):
                            self.seen_ids.add(tid)
                            self.passed += 1
                else:
                    self.frames_in_zone[tid] = 0

                color = (0, 255, 0) if inside else (0, 0, 255)
                thickness = 2 if should_detect else 1
                cv2.rectangle(process_frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
                cv2.putText(process_frame, str(tid), (box[0], box[1]-5), 0, 0.6, color, thickness)

            if frame_idx % 5 == 0:
                t_end = time.time()
                dt_ms = (t_end - t_start) * 1000
                mem = self.process.memory_info().rss / 1024**2
                self.root.after(0, self.update_perf_ui, dt_ms, mem)
                self.root.after(0, self.update_stats_ui)

            img_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
            
            def update_image():
                if not self.play: return
                self.lbl_vid.config(image=img_tk)
                self.lbl_vid.image = img_tk
                
            self.root.after(0, update_image)
            frame_idx += 1

            t_process_end = time.time()
            elapsed = t_process_end - t_start
            wait_time = frame_dur - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()