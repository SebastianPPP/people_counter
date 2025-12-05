import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import torch
from collections import defaultdict, deque

class LowLevelTracker:
    """
    Simple Re-ID enhancement for your existing tracker
    Handles basic cases where people temporarily disappear
    """
    def __init__(self, max_disappeared_frames=15):
        self.max_disappeared_frames = max_disappeared_frames
        self.person_registry = {}  # {unique_id: person_info}
        self.track_to_unique = {}  # {track_id: unique_id}
        self.disappeared_trackers = {}  # {track_id: frames_disappeared}
        self.next_unique_id = 1
        
        # Simple feature storage (position + size history)
        self.position_history = defaultdict(lambda: deque(maxlen=5))
        
    def get_bbox_features(self, bbox):
        """Extract simple features from bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        area = width * height
        
        return np.array([center_x, center_y, width, height, aspect_ratio, area])
    
    def calculate_distance(self, features1, features2):
        """Calculate distance between two feature vectors"""
        # Normalize position by frame size (assuming 640x360)
        f1_norm = features1.copy()
        f2_norm = features2.copy()
        f1_norm[0] /= 640  # normalize x
        f1_norm[1] /= 360  # normalize y
        f2_norm[0] /= 640
        f2_norm[1] /= 360
        
        # Weight different features
        weights = np.array([1.0, 1.0, 0.5, 0.5, 0.3, 0.7])  # pos, pos, w, h, aspect, area
        
        return np.linalg.norm((f1_norm - f2_norm) * weights)
    
    def find_best_match_for_reappeared_track(self, track_id, current_features):
        """Find best match among disappeared persons for a new track"""
        best_match_id = None
        best_distance = float('inf')
        
        # Check recently disappeared tracks
        for disappeared_track_id, frames_gone in list(self.disappeared_trackers.items()):
            if disappeared_track_id in self.track_to_unique:
                unique_id = self.track_to_unique[disappeared_track_id]
                
                if unique_id in self.person_registry:
                    # Get last known features
                    last_features = self.person_registry[unique_id]['last_features']
                    distance = self.calculate_distance(current_features, last_features)
                    
                    # Consider match if distance is reasonable and not too much time passed
                    if distance < best_distance and distance < 50 and frames_gone < self.max_disappeared_frames:
                        best_distance = distance
                        best_match_id = unique_id
        
        return best_match_id, best_distance
    
    def update(self, detections):
        """
        Update tracking with simple re-ID
        detections: list of (x1, y1, x2, y2, track_id) tuples
        """
        current_track_ids = set()
        
        # Process current detections
        for x1, y1, x2, y2, track_id in detections:
            if track_id == -1:
                continue
                
            current_track_ids.add(track_id)
            current_features = self.get_bbox_features((x1, y1, x2, y2))
            
            # Check if this is a new track
            if track_id not in self.track_to_unique:
                # Try to match with recently disappeared person
                matched_unique_id, distance = self.find_best_match_for_reappeared_track(
                    track_id, current_features)
                
                if matched_unique_id is not None:
                    # Re-assign disappeared person to new track
                    self.track_to_unique[track_id] = matched_unique_id
                    
                    # Update person info
                    self.person_registry[matched_unique_id]['last_features'] = current_features
                    self.person_registry[matched_unique_id]['total_detections'] += 1
                    
                    # Remove from disappeared list (clean up old track)
                    old_tracks_to_remove = []
                    for old_track_id, unique_id in self.track_to_unique.items():
                        if unique_id == matched_unique_id and old_track_id != track_id:
                            old_tracks_to_remove.append(old_track_id)
                    
                    for old_track_id in old_tracks_to_remove:
                        if old_track_id in self.disappeared_trackers:
                            del self.disappeared_trackers[old_track_id]
                        if old_track_id in self.track_to_unique:
                            del self.track_to_unique[old_track_id]
                    
                    print(f"Re-identified person {matched_unique_id}: old track -> new track {track_id}")
                else:
                    # Create new person
                    unique_id = self.next_unique_id
                    self.next_unique_id += 1
                    
                    self.track_to_unique[track_id] = unique_id
                    self.person_registry[unique_id] = {
                        'first_seen_frame': 0,  # You might want to track frame numbers
                        'total_detections': 1,
                        'last_features': current_features,
                        'status': 'active',
                        'counted': False  # whether this unique person has been counted for central-region pass
                    }
                    
                    print(f"New person detected: unique_id {unique_id}, track_id {track_id}")
            else:
                # Update existing person
                unique_id = self.track_to_unique[track_id]
                if unique_id in self.person_registry:
                    self.person_registry[unique_id]['last_features'] = current_features
                    self.person_registry[unique_id]['total_detections'] += 1
                    self.person_registry[unique_id]['status'] = 'active'
                    # ensure counted flag exists (in case of older registry entries)
                    if 'counted' not in self.person_registry[unique_id]:
                        self.person_registry[unique_id]['counted'] = False
            
            # Update position history
            if track_id in self.track_to_unique:
                unique_id = self.track_to_unique[track_id]
                self.position_history[unique_id].append(current_features[:2])  # just center position
            
            # Remove from disappeared if it was there
            if track_id in self.disappeared_trackers:
                del self.disappeared_trackers[track_id]
        
        # Handle disappeared tracks
        all_known_tracks = set(self.track_to_unique.keys())
        disappeared_tracks = all_known_tracks - current_track_ids
        
        for track_id in disappeared_tracks:
            self.disappeared_trackers[track_id] = self.disappeared_trackers.get(track_id, 0) + 1
            
            # Mark person as temporarily disappeared
            if track_id in self.track_to_unique:
                unique_id = self.track_to_unique[track_id]
                if unique_id in self.person_registry:
                    self.person_registry[unique_id]['status'] = 'disappeared'
        
        # Clean up tracks that have been gone too long
        tracks_to_remove = []
        for track_id, frames_disappeared in self.disappeared_trackers.items():
            if frames_disappeared > self.max_disappeared_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.track_to_unique:
                unique_id = self.track_to_unique[track_id]
                if unique_id in self.person_registry:
                    self.person_registry[unique_id]['status'] = 'lost'
                del self.track_to_unique[track_id]
            del self.disappeared_trackers[track_id]
        
        # Return current stats
        active_people = sum(1 for p in self.person_registry.values() if p['status'] == 'active')
        total_people_seen = len(self.person_registry)
        
        return {
            'active_count': active_people,
            'total_unique': total_people_seen,
            'track_to_unique': self.track_to_unique.copy(),
            'person_registry': self.person_registry.copy()
        }

class ImprovedPeopleCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("People counter")
        self.root.geometry("1000x700")
        self.root.config(bg="#1e1e1e")

        # YOLOv8 model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt").to(self.device)
        
        # We do not use unique-person registry anymore.
        # Track by tracker `track_id` only: keep short history per track and a set of already-counted tracks.
        self.track_history = defaultdict(lambda: deque(maxlen=5))
        self.counted_tracks = set()
        # Per-track speed history (pixels/sec) for average speed
        self.track_speeds = defaultdict(lambda: deque(maxlen=5))
        # Per-track time-in-central counters (frames)
        self.track_current_in_region = defaultdict(int)
        self.track_total_in_region = defaultdict(int)

        # Video
        self.cap = None
        self.playing = False
        self.video_path = None
        self.frame_image = None

        # Statistics
        self.passed_count = 0
        self.frame_count = 0

        # Central region as fraction of frame (centered rectangle)
        # You can adjust these values: (x_frac, y_frac, w_frac, h_frac)
        self.central_region = (0.5, 0, 0.3, 1)  # centered rectangle covering middle 30% width, 100% height

        self.setup_gui()

    def setup_gui(self):
        # Header
        tk.Label(self.root, text="People counter",
                 font=("Arial", 18, "bold"), bg="#1e1e1e", fg="white").pack(pady=10)

        # Control buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Choose video", command=self.choose_video, width=15,
                  bg="#007acc", fg="white", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        
        tk.Button(btn_frame, text="Start", command=self.start_video, width=10,
                  bg="#28a745", fg="white", font=("Arial", 12)).grid(row=0, column=1, padx=5)
        
        tk.Button(btn_frame, text="Stop", command=self.stop_video, width=10,
                  bg="#dc3545", fg="white", font=("Arial", 12)).grid(row=0, column=2, padx=5)
        
        tk.Button(btn_frame, text="Reset", command=self.reset_tracker, width=10,
                  bg="#ffc107", fg="black", font=("Arial", 12)).grid(row=0, column=3, padx=5)

        # Video display
        self.video_label = tk.Label(self.root, bg="#000000", width=640, height=360)
        self.video_label.pack(pady=10)

        # Statistics
        stats_frame = tk.Frame(self.root, bg="#2d2d2d", relief="raised", bd=2)
        stats_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(stats_frame, text="Statistics:", font=("Arial", 14, "bold"),
                bg="#2d2d2d", fg="white").pack(pady=5)
        
        self.frame_label = tk.Label(stats_frame, text="Frame: 0",
                       font=("Arial", 12), bg="#2d2d2d", fg="#aaaaaa")
        self.frame_label.pack(pady=2)

        # Passed counter label
        self.passed_label = tk.Label(stats_frame, text="Passed people: 0",
                         font=("Arial", 12), bg="#2d2d2d", fg="#ffffff")
        self.passed_label.pack(pady=2)

        # Average time in central region (seconds)
        self.avg_time_label = tk.Label(stats_frame, text="Avg time (s): 0.0",
                           font=("Arial", 12), bg="#2d2d2d", fg="#ffffff")
        self.avg_time_label.pack(pady=2)

        # Average speed (px/s)
        self.avg_speed_label = tk.Label(stats_frame, text="Avg speed (px/s): 0.0",
                        font=("Arial", 12), bg="#2d2d2d", fg="#ffffff")
        self.avg_speed_label.pack(pady=2)

    def reset_tracker(self):
        """Reset the tracking system"""
        # Clear per-track history and counted tracks
        self.track_history.clear()
        self.counted_tracks.clear()
        self.track_speeds.clear()
        self.track_current_in_region.clear()
        self.track_total_in_region.clear()
        self.current_people = 0
        self.passed_count = 0
        self.frame_count = 0
        self.update_stats_display()

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            messagebox.showinfo("Video selected", f"Path: {path}")

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No video", "Please select a video file.")
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return
        # store fps for speed/time calculations
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.playing = True
        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_video(self):
        self.playing = False
        if self.cap:
            self.cap.release()

    def process_video(self):
        while self.playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.resize(frame, (640, 360))

            # YOLOv8 detection & tracking
            try:
                results = self.model.track(frame, persist=True, verbose=False)[0]
            except Exception:
                results = self.model(frame, verbose=False)[0]

            # Extract detections
            detections = []
            for box in results.boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1
                    detections.append((x1, y1, x2, y2, track_id))

            # Determine central region in pixel coords
            h, w = frame.shape[:2]
            rx, ry, rw, rh = self.central_region
            cx1 = int((rx) * w)
            cy1 = int((ry) * h)
            cx2 = int((rx + rw) * w)
            cy2 = int((ry + rh) * h)

            # For each detection, compute centroid, update per-track history and check transitions
            inside_tracks = set()

            for x1, y1, x2, y2, track_id in detections:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # update track history
                if track_id is not None and track_id != -1:
                    hist = self.track_history[track_id]
                    hist.append((cx, cy))

                    # compute instantaneous speed (pixels/sec) from last two positions
                    if len(hist) >= 2:
                        (px_prev, py_prev) = hist[-2]
                        dist = ((cx - int(px_prev))**2 + (cy - int(py_prev))**2) ** 0.5
                        speed_px_s = dist * (self.fps if hasattr(self, 'fps') else 30.0)
                        self.track_speeds[track_id].append(speed_px_s)

                # whether centroid is inside central region
                inside = (cx1 <= cx <= cx2) and (cy1 <= cy <= cy2)
                prev_inside = False
                if track_id is not None and track_id != -1:
                    # previous position (if exists)
                    prev_list = list(self.track_history.get(track_id, []))
                    if len(prev_list) >= 2:
                        px_prev, py_prev = prev_list[-2]
                        prev_inside = (cx1 <= int(px_prev) <= cx2) and (cy1 <= int(py_prev) <= cy2)

                if inside and track_id is not None and track_id != -1:
                    inside_tracks.add(track_id)

                    # increment current in-region counter
                    self.track_current_in_region[track_id] += 1

                    # If track hasn't been counted yet, check whether it came from outside
                    if track_id not in self.counted_tracks:
                        was_outside = False
                        hist = list(self.track_history.get(track_id, []))
                        # exclude current position when checking previous positions
                        prev_positions = hist[:-1] if len(hist) > 0 else []
                        for (px, py) in prev_positions:
                            if not ((cx1 <= int(px) <= cx2) and (cy1 <= int(py) <= cy2)):
                                was_outside = True
                                break

                        # If we saw previous position outside, count it
                        if was_outside:
                            self.counted_tracks.add(track_id)
                            self.passed_count += 1
                            print(f"Track {track_id} counted as passed. Total: {self.passed_count}")
                else:
                    # if previously was inside (we count consecutive inside frames), finalize and add to total
                    if track_id is not None and track_id != -1 and self.track_current_in_region.get(track_id, 0) > 0:
                        self.track_total_in_region[track_id] += self.track_current_in_region[track_id]
                        self.track_current_in_region[track_id] = 0

            # Draw detections for tracks inside central region only, show time and clothing color
            for x1, y1, x2, y2, track_id in detections:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                inside = (cx1 <= cx <= cx2) and (cy1 <= cy <= cy2)
                if not inside:
                    continue

                # Choose color based on track ID for bbox
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
                         (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 165, 0)]
                color = colors[track_id % len(colors)] if (track_id is not None and track_id != -1) else (0, 255, 0)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Compute clothing color (sample upper half of bbox)
                try:
                    hbox = max(1, int((y2 - y1) * 0.5))
                    crop_y1 = max(0, y1)
                    crop_y2 = min(y2, y1 + hbox)
                    crop_x1 = max(0, x1)
                    crop_x2 = min(w - 1, x2)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if crop.size > 0:
                        avg_color = tuple(map(int, crop.reshape(-1, 3).mean(axis=0)))
                    else:
                        avg_color = (200, 200, 200)
                except Exception:
                    avg_color = (200, 200, 200)

                # Draw a small color swatch for clothing color to the right of the bbox
                sw_x1 = min(x2 + 5, w - 25)
                sw_y1 = max(y1, 0)
                sw_x2 = sw_x1 + 20
                sw_y2 = sw_y1 + 14
                cv2.rectangle(frame, (sw_x1, sw_y1), (sw_x2, sw_y2), avg_color, -1)

                # Compute total time in region (seconds)
                total_frames = self.track_total_in_region.get(track_id, 0) + self.track_current_in_region.get(track_id, 0)
                time_sec = (total_frames / (self.fps if hasattr(self, 'fps') else 30.0)) if track_id is not None and track_id != -1 else 0.0

                # Draw track ID and time next to bbox
                label = f"T:{track_id} {time_sec:.1f}s"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # draw central region rectangle
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 128, 255), 2)
            cv2.putText(frame, 'Central region', (cx1 + 5, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,255), 2)

            # Update statistics (frame, passed count, average time, average speed)
            self.passed_label.config(text=f"Passed people: {self.passed_count}")
            self.frame_label.config(text=f"Frame: {self.frame_count}")

            # Average time in central region (include current and total frames for any track that had >0)
            total_times = []
            for tid in set(list(self.track_total_in_region.keys()) + list(self.track_current_in_region.keys())):
                tot = self.track_total_in_region.get(tid, 0) + self.track_current_in_region.get(tid, 0)
                if tot > 0:
                    total_times.append(tot / (self.fps if hasattr(self, 'fps') else 30.0))
            avg_time = float(np.mean(total_times)) if len(total_times) > 0 else 0.0
            self.avg_time_label.config(text=f"Avg time (s): {avg_time:.1f}")

            # Average speed across tracks (use mean of per-track mean speeds)
            speed_means = []
            for tid, dq in self.track_speeds.items():
                if len(dq) > 0:
                    speed_means.append(float(np.mean(dq)))
            avg_speed = float(np.mean(speed_means)) if len(speed_means) > 0 else 0.0
            self.avg_speed_label.config(text=f"Avg speed (px/s): {avg_speed:.1f}")

            self.update_gui(frame)

        self.stop_video()

    def update_stats_display(self):
        self.frame_label.config(text=f"Frame: {self.frame_count}")
        self.passed_label.config(text=f"Passed people: {self.passed_count}")

        # Also update averages when called (e.g., after reset)
        total_times = []
        for tid in set(list(self.track_total_in_region.keys()) + list(self.track_current_in_region.keys())):
            tot = self.track_total_in_region.get(tid, 0) + self.track_current_in_region.get(tid, 0)
            if tot > 0:
                total_times.append(tot / (self.fps if hasattr(self, 'fps') else 30.0))
        avg_time = float(np.mean(total_times)) if len(total_times) > 0 else 0.0
        self.avg_time_label.config(text=f"Avg time (s): {avg_time:.1f}")

        speed_means = []
        for tid, dq in self.track_speeds.items():
            if len(dq) > 0:
                speed_means.append(float(np.mean(dq)))
        avg_speed = float(np.mean(speed_means)) if len(speed_means) > 0 else 0.0
        self.avg_speed_label.config(text=f"Avg speed (px/s): {avg_speed:.1f}")

    def update_gui(self, frame):
        self.update_stats_display()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.frame_image = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.frame_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedPeopleCounterGUI(root)
    root.mainloop()