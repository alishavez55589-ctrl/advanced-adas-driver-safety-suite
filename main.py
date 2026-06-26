import cv2
import time
import math
import numpy as np
from threading import Thread
import os
import platform
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ReliableVideoCapture:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.thread = None

    def start(self):
        if self.started: 
            return self
        self.started = True
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
                break
            self.frame = frame

    def read(self): 
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened(): 
            self.cap.release()

class UltimateSafetySuiteEngine:
    def __init__(self, face_model='face_landmarker.task', pose_model='pose_landmarker_full.task'):
        base_face_opts = python.BaseOptions(model_asset_path=face_model)
        self.face_detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_face_opts, num_faces=1)
        )
        
        base_pose_opts = python.BaseOptions(model_asset_path=pose_model)
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base_pose_opts, num_poses=1)
        )
        
        # State Tracking Matrix Timers
        self.eye_closed_start_time = None
        self.yawn_start_time = None
        self.distract_start_time = None
        
        # Rolling Temporal Windows for Distraction Parsing
        self.horizontal_distraction_events = []
        self.vertical_distraction_events = [] 
        
        # System State Flags
        self.is_drowsy = False
        self.is_yawning = False
        self.is_distracted = False
        self.seatbelt_fastened = True 
        
        self.distraction_type = "" 
        self.recovery_start_time = None
        
        # Configured Cooldowns & Thresholds
        self.cooldown_duration = 2.5 
        self.alert_trigger_threshold = 5.0  # 5 Seconds Delay Requirement

    @staticmethod
    def calculate_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_ratio(self, coords, vertical_1, vertical_2, horizontal_1, horizontal_2):
        v = self.calculate_distance(coords[vertical_1], coords[vertical_2])
        h = self.calculate_distance(coords[horizontal_1], coords[horizontal_2])
        return v / h if h != 0 else 0.0

    def check_seatbelt_strap(self, frame, p1, p2):
        try:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.line(mask, p1, p2, 255, 8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            values = gray[mask == 255]
            if len(values) == 0: 
                return True
            return float(np.std(values)) > 12.0
        except:
            return True

    def trigger_audio_alarm(self):
        """System level hardware beep engine - No external file dependencies"""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 200)
            else:
                print("\a", end="", flush=True)
        except:
            pass

    def process_frame(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        face_result = self.face_detector.detect(mp_image)
        pose_result = self.pose_detector.detect(mp_image)
        
        status = "DRIVER ACTIVE"
        color = (0, 255, 0)
        current_ear, current_mar = -1.0, -1.0
        head_horizontal_ratio, head_vertical_ratio = 1.0, 1.0

        if face_result.face_landmarks:
            landmarks = face_result.face_landmarks[0]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            
            current_ear = (self.get_ratio(coords, 159, 145, 33, 133) + self.get_ratio(coords, 386, 374, 362, 263)) / 2.0
            current_mar = self.get_ratio(coords, 13, 14, 78, 308)
            
            head_horizontal_ratio = self.calculate_distance(coords[4], coords[234]) / self.calculate_distance(coords[4], coords[454]) if self.calculate_distance(coords[4], coords[454]) != 0 else 1.0
            head_vertical_ratio = self.calculate_distance(coords[4], coords[152]) / self.calculate_distance(coords[4], coords[10]) if self.calculate_distance(coords[4], coords[10]) != 0 else 1.0
            
            for idx in [159, 145, 33, 133, 386, 374, 362, 263, 13, 14, 4]:
                cv2.circle(frame, coords[idx], 2, (0, 255, 255), -1)

        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            l_sh = (int(pose_lms[11].x * w), int(pose_lms[11].y * h))
            r_sh = (int(pose_lms[12].x * w), int(pose_lms[12].y * h))
            
            cv2.circle(frame, l_sh, 4, (255, 0, 0), -1)
            cv2.circle(frame, r_sh, 4, (255, 0, 0), -1)
            
            chest_center = (int((l_sh[0] + r_sh[0]) / 2), int((l_sh[1] + r_sh[1]) / 2) + 65)
            self.seatbelt_fastened = self.check_seatbelt_strap(frame, l_sh, chest_center)
            
            line_color = (0, 255, 0) if self.seatbelt_fastened else (0, 0, 255)
            cv2.line(frame, l_sh, chest_center, line_color, 2)

        # State Engine Core Mathematics Evaluation
        current_time = time.time()

        # Fatigue Parsing (Threshold: 5s)
        if current_ear != -1.0 and current_ear < 0.24:
            if self.eye_closed_start_time is None: 
                self.eye_closed_start_time = current_time
            if (current_time - self.eye_closed_start_time) > self.alert_trigger_threshold: 
                self.is_drowsy, self.recovery_start_time = True, None
                self.trigger_audio_alarm()
        else: 
            self.eye_closed_start_time = None

        # Yawn Parsing (Threshold: 5s)
        if current_mar != -1.0 and current_mar > 0.62:
            if self.yawn_start_time is None: 
                self.yawn_start_time = current_time
            if (current_time - self.yawn_start_time) > self.alert_trigger_threshold: 
                self.is_yawning, self.recovery_start_time = True, None
                self.trigger_audio_alarm()
        else: 
            self.yawn_start_time = None

        # Spatial Head Orientations Analytics (Threshold: 5s)
        side_look = (head_horizontal_ratio < 0.52 or head_horizontal_ratio > 1.95)
        down_look = (head_vertical_ratio < 0.85)
        
        if side_look or down_look:
            if self.distract_start_time is None: 
                self.distract_start_time = current_time
            if (current_time - self.distract_start_time) > self.alert_trigger_threshold: 
                self.is_distracted = True
                self.distraction_type = "ALERT: DISTRACTED!" if side_look else "ALERT: MOBILE USE!"
                self.recovery_start_time = None
                self.trigger_audio_alarm()
        else: 
            self.distract_start_time = None

        # Event Windows Sync Management
        if side_look and (not self.horizontal_distraction_events or (current_time - self.horizontal_distraction_events[-1]) > 0.8):
            self.horizontal_distraction_events.append(current_time)
        if down_look and (not self.vertical_distraction_events or (current_time - self.vertical_distraction_events[-1]) > 0.8):
            self.vertical_distraction_events.append(current_time)
            
        self.horizontal_distraction_events = [t for t in self.horizontal_distraction_events if current_time - t <= 10.0]
        self.vertical_distraction_events = [t for t in self.vertical_distraction_events if current_time - t <= 10.0]

        if len(self.vertical_distraction_events) >= 5: 
            self.is_distracted, self.distraction_type, self.recovery_start_time = True, "ALERT: MOBILE USE!", None
            self.trigger_audio_alarm()
        elif len(self.horizontal_distraction_events) >= 5: 
            self.is_distracted, self.distraction_type, self.recovery_start_time = True, "ALERT: DISTRACTED!", None
            self.trigger_audio_alarm()

        # Cooldown State Matrix
        any_alert = self.is_drowsy or self.is_yawning or self.is_distracted
        all_clear = (current_ear >= 0.24 and current_mar <= 0.62 and not side_look and not down_look)
        
        if all_clear and any_alert:
            if self.recovery_start_time is None: 
                self.recovery_start_time = current_time
            cooldown_left = self.cooldown_duration - (current_time - self.recovery_start_time)
            if cooldown_left <= 0:
                self.is_drowsy = self.is_yawning = self.is_distracted = False
                self.recovery_start_time = None
                self.horizontal_distraction_events.clear()
                self.vertical_distraction_events.clear()
            else: 
                status, color = f"RECOVERING... ({int(cooldown_left)+1}s)", (0, 165, 255)

        # UI Priority Routing Configuration
        if self.recovery_start_time is None:
            if self.is_drowsy: 
                status, color = "ALERT: DROWSY DRIVER!", (0, 0, 255)
            elif self.is_yawning: 
                status, color = "ALERT: DRIVER YAWNING!", (0, 69, 255)
            elif self.is_distracted: 
                status, color = self.distraction_type, (0, 128, 255)
            elif not self.seatbelt_fastened: 
                status, color = "ALERT: FASTEN SEATBELT!", (0, 0, 255)

        if (self.is_drowsy or self.is_yawning or self.is_distracted or not self.seatbelt_fastened) and self.recovery_start_time is None:
            cv2.rectangle(frame, (0, 0), (w, h), color, 12) 

        return frame, status, color, current_ear, current_mar, head_horizontal_ratio, head_vertical_ratio

if __name__ == "__main__":
    vs = ReliableVideoCapture().start()
    time.sleep(1.0) 
    engine = UltimateSafetySuiteEngine()
    prev_time = 0
    
    while True:
        frame = vs.read()
        if frame is None: 
            continue
            
        h, w, _ = frame.shape
        frame, alert_status, ui_color, live_ear, live_mar, live_horiz, live_vert = engine.process_frame(frame)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # UI Renderer Engine Panel
        cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {int(fps)}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        ear_lbl = f"EAR: {live_ear:.2f}" if live_ear != -1.0 else "EAR: LOST"
        mar_lbl = f"MAR: {live_mar:.2f}" if live_mar != -1.0 else "MAR: LOST"
        hz_lbl = f"H-ALN: {live_horiz:.2f}"
        vt_lbl = f"V-ALN: {live_vert:.2f}"
        
        cv2.putText(frame, ear_lbl, (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, mar_lbl, (100, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(frame, hz_lbl, (185, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, vt_lbl, (285, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.putText(frame, alert_status, (w - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
        
        cv2.imshow("Advanced ADAS Safety Suite Dashboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    vs.stop()
    cv2.destroyAllWindows()