import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
from collections import defaultdict, deque

class LowLevelTracker:
    def __init__(self, max_lost=30, iou_thresh=0.3):
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        
        # Structure: {id: {'box': [x1,y1,x2,y2], 'feats': vec, 'status': str, 'lost_cnt': int}}
        self.tracks = {} 
        self.next_id = 1
        
        # Pos history for linear prediction
        self.pos_hist = defaultdict(lambda: deque(maxlen=5))

    def get_feats(self, box, shape):
        x1, y1, x2, y2 = box
        w, h = shape
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw, bh = x2-x1, y2-y1
        # Normalize to 0-1 
        return np.array([cx/w, cy/h, bw/w, bh/h, (bw/bh if bh else 1.0)], dtype=np.float32)

    def calc_iou(self, boxesA, boxesB):
        # Enforce float32 for low-level optimization
        boxesA = np.array(boxesA, dtype=np.float32)
        boxesB = np.array(boxesB, dtype=np.float32)
        
        if len(boxesA) == 0 or len(boxesB) == 0:
            return np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)

        xA = np.maximum(boxesA[:, None, 0], boxesB[None, :, 0])
        yA = np.maximum(boxesA[:, None, 1], boxesB[None, :, 1])
        xB = np.minimum(boxesA[:, None, 2], boxesB[None, :, 2])
        yB = np.minimum(boxesA[:, None, 3], boxesB[None, :, 3])

        # IoU
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
            
            # Calc velocity vector from last 2 frames
            if len(hist) >= 2:
                dx = hist[-1][0] - hist[-2][0]
                dy = hist[-1][1] - hist[-2][1]
            
            # Apply prediction
            preds.append((box[0]+dx, box[1]+dy, box[2]+dx, box[3]+dy))
            ids.append(tid)
        return preds, ids

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
            
            # Re-ID based on visual features distance
            best_id, best_dist = None, float('inf')
            for tid, info in self.tracks.items():
                if info['status'] == 'lost':
                    dist = np.linalg.norm(feats - info['feats'])
                    if dist < best_dist and dist < 0.2: 
                        best_dist = dist
                        best_id = tid
            
            if best_id: # Recovered
                tid = best_id
                self.tracks[tid].update({'box': box, 'feats': feats, 'status': 'active', 'lost_cnt': 0})
            else: # New Track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': box, 'feats': feats, 'status': 'active', 'lost_cnt': 0}
            
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            self.pos_hist[tid].append((cx, cy))

        # handle lost
        for tid in unmatched_tracks:
            self.tracks[tid]['status'] = 'lost'
            self.tracks[tid]['lost_cnt'] += 1

        to_del = [tid for tid, info in self.tracks.items() if info['lost_cnt'] > self.max_lost]
        for tid in to_del: del self.tracks[tid]

        return {tid: i['box'] for tid, i in self.tracks.items() if i['status'] == 'active'}

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("People Counter")
        self.root.config(bg="#1e1e1e")

        # Models
        self.model = YOLO("yolov8n.pt") 
        self.tracker = LowLevelTracker(max_lost=30) 
        
        # Stats
        self.passed = 0
        self.seen_ids = set()
        self.frames_in_zone = defaultdict(int) 
        self.speeds = [] 
        
        self.zone_rel = (0.5, 0, 0.3, 1)

        # Video
        self.cap = None
        self.play = False
        self.path = None
        self.fps = 30.0

        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Counter", bg="#1e1e1e", fg="white", font=("Arial", 16)).pack(pady=10)
        
        btns = tk.Frame(self.root, bg="#1e1e1e")
        btns.pack()
        tk.Button(btns, text="Load", command=self.load, bg="#007acc", fg="white", width=10).grid(row=0, column=0, padx=5)
        tk.Button(btns, text="Start", command=self.start, bg="#28a745", fg="white", width=10).grid(row=0, column=1, padx=5)
        tk.Button(btns, text="Stop", command=self.stop, bg="#dc3545", fg="white", width=10).grid(row=0, column=2, padx=5)
        tk.Button(btns, text="Reset", command=self.reset, bg="#ffc107", fg="black", width=10).grid(row=0, column=3, padx=5)

        self.lbl_vid = tk.Label(self.root, bg="black")
        self.lbl_vid.pack(pady=10)

        self.lbl_stats = tk.Label(self.root, text="Passed: 0 | Avg Speed: 0.0 px/s", bg="#1e1e1e", fg="white", font=("Arial", 12))
        self.lbl_stats.pack(pady=5)

    def load(self):
        f = filedialog.askopenfilename()
        if f: self.path = f

    def start(self):
        if not self.path: return
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.play = True
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self): self.play = False
    
    def reset(self):
        self.tracker = LowLevelTracker()
        self.passed = 0
        self.seen_ids.clear()
        self.frames_in_zone.clear()
        self.speeds.clear()
        self.update_stats()

    def update_stats(self):
        avg_v = np.mean(self.speeds) if self.speeds else 0.0
        self.lbl_stats.config(text=f"Passed: {self.passed} | Avg Speed: {avg_v:.1f} px/s")

    def loop(self):
        while self.play and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 360))
            h, w = frame.shape[:2]

            # detect
            res = self.model.predict(frame, verbose=False)[0]
            # Convert tensors to int list
            dets = [tuple(b.xyxy[0].cpu().numpy().astype(int)) 
                    for b in res.boxes if int(b.cls[0]) == 0] 

            # track
            tracks = self.tracker.update(dets, (w, h))


            rx, ry, rw, rh = self.zone_rel
            zx1, zx2 = int((rx)*w), int((rx+rw)*w)
            zone_s = zx2 - zx1 

            for tid, box in tracks.items():
                cx = (box[0] + box[2]) // 2
                inside = zx1 <= cx <= zx2

                if inside:
                    self.frames_in_zone[tid] += 1
                    # Count logic
                    if tid not in self.seen_ids:
                        hist = self.tracker.pos_hist[tid]
                        # Check entry from outside
                        if len(hist) > 1 and not (zx1 <= hist[-2][0] <= zx2):
                            self.seen_ids.add(tid)
                            self.passed += 1
                else:
                    t_frames = self.frames_in_zone[tid]
                    if t_frames > 0:
                        t_sec = t_frames / self.fps
                        # Filter noise (> 0.5s duration)
                        if t_sec > 0.5:
                            v = zone_s / t_sec 
                            self.speeds.append(v)
                        self.frames_in_zone[tid] = 0

                # Draw
                col = (0, 255, 0) if inside else (0, 0, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, 2)
                cv2.putText(frame, f"ID:{tid}", (box[0], box[1]-5), 0, 0.5, col, 2)

            # Draw Zone
            cv2.rectangle(frame, (zx1, 0), (zx2, h), (255, 255, 0), 2)
            
            self.update_stats()
            # Convert to TK
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.lbl_vid.config(image=img)
            self.lbl_vid.image = img

        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()