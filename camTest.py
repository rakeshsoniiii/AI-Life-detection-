

import cv2
import numpy as np
import time
import random
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List
from collections import deque

# ─── Attempt to import Ultralytics YOLO ───────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed. Running in DEMO mode (simulated detections).")

# ─── Logging Configuration ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s → %(message)s",
    datefmt="%H:%M:%S"
)

# =============================================================================
# ENUMERATIONS & DATA STRUCTURES
# =============================================================================

class DisasterMode(Enum):
    NONE       = "STANDBY"
    EARTHQUAKE = "EARTHQUAKE"
    FIRE       = "FIRE"
    FLOOD      = "FLOOD"

class RoverMovementState(Enum):
    FORWARD   = "MOVING FORWARD"
    STOPPED   = "STOPPED"
    AVOIDING  = "OBSTACLE AVOIDANCE"
    RETURNING = "RETURNING TO BASE"
    EMERGENCY = "EMERGENCY STOP"

@dataclass
class DetectionResult:
    """Structured output from the DetectionModule."""
    survivor_detected:  bool        = False
    obstacle_detected:  bool        = False
    detections:         List[dict]  = field(default_factory=list)
    annotated_frame:    Optional[np.ndarray] = None

@dataclass
class SensorData:
    """Fused sensor data packet."""
    lidar_distance_cm:   float = 150.0
    obstacle_confirmed:  bool  = False
    timestamp:           float = 0.0

@dataclass
class SystemState:
    """
    Central shared state — single source of truth for the entire system.
    Passed by reference to all modules for coordination.
    """
    # ── Fail-safe flags
    communication_lost:  bool  = False
    emergency_stop:      bool  = False

    # ── Battery simulation
    battery_level:       float = 100.0
    battery_drain_rate:  float = 0.012        # % per frame at ~30fps → ~2min runtime

    # ── Disaster mode
    disaster_mode:       DisasterMode = DisasterMode.NONE

    # ── Runtime bookkeeping
    frame_count:         int   = 0
    session_start:       float = field(default_factory=time.time)
    fps:                 float = 0.0

    # ── Latest detection / sensor / decision snapshots (filled by modules)
    detection:           DetectionResult   = field(default_factory=DetectionResult)
    sensor_data:         SensorData        = field(default_factory=SensorData)
    current_action:      str               = "INITIALIZING"
    movement_state:      RoverMovementState = RoverMovementState.STOPPED

    # ── FPS ring-buffer
    fps_history:         deque = field(default_factory=lambda: deque(maxlen=30))
    last_frame_time:     float = field(default_factory=time.time)

    def tick(self):
        """Called once per frame to update battery, FPS, and frame counter."""
        now = time.time()
        dt = now - self.last_frame_time
        if dt > 0:
            self.fps_history.append(1.0 / dt)
            self.fps = sum(self.fps_history) / len(self.fps_history)
        self.last_frame_time = now
        self.frame_count += 1

        # Battery drain — only when not in emergency stop
        if not self.emergency_stop:
            self.battery_level = max(0.0, self.battery_level - self.battery_drain_rate)


# =============================================================================
# MODULE 1 — CAMERA MODULE
# =============================================================================

class CameraModule:
    """
    Handles webcam capture and basic pre-processing.
    Abstraction layer so the rest of the system is camera-agnostic.
    """

    CAPTURE_WIDTH  = 1280
    CAPTURE_HEIGHT = 720
    YOLO_SIZE      = 640       # YOLO inference resolution

    def __init__(self, camera_index: int = 0):
        self._logger = logging.getLogger("CameraModule")
        self._cap: Optional[cv2.VideoCapture] = None
        self._camera_index = camera_index
        self._initialized = False

    def initialize(self) -> bool:
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._logger.error("Failed to open webcam (index %d).", self._camera_index)
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._initialized = True
        self._logger.info("Camera initialized — %dx%d", self.CAPTURE_WIDTH, self.CAPTURE_HEIGHT)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (success, frame). Frame is the raw capture at full resolution."""
        if not self._initialized:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            self._logger.warning("Frame capture failed.")
            return False, None
        return True, frame

    def get_yolo_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to YOLO input size."""
        return cv2.resize(frame, (self.YOLO_SIZE, self.YOLO_SIZE))

    def release(self):
        if self._cap:
            self._cap.release()
        self._logger.info("Camera released.")


# =============================================================================
# MODULE 2 — DETECTION MODULE (YOLOv8)
# =============================================================================

class DetectionModule:
    """
    Runs YOLOv8 inference to detect survivors and obstacles.
    Falls back to randomized demo detections if YOLO is unavailable.
    """

    # COCO classes treated as "survivor"
    SURVIVOR_CLASSES = {"person"}

    # COCO classes treated as disaster "debris/obstacle"
    OBSTACLE_CLASSES = {
        "chair", "car", "truck", "motorcycle", "bicycle",
        "backpack", "suitcase", "bench", "dining table",
        "tv", "laptop", "refrigerator", "couch", "bed"
    }

    CONFIDENCE_THRESHOLD = 0.40

    # Color palette (BGR)
    COLOR_SURVIVOR  = (0,  255, 100)
    COLOR_OBSTACLE  = (0,  80,  255)
    COLOR_OTHER     = (200, 200, 200)

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._logger = logging.getLogger("DetectionModule")
        self._model = None
        self._model_path = model_path
        self._demo_mode = not YOLO_AVAILABLE

    def initialize(self) -> bool:
        if self._demo_mode:
            self._logger.warning("DEMO MODE — using simulated detections.")
            return True
        try:
            self._logger.info("Loading YOLO model: %s", self._model_path)
            self._model = YOLO(self._model_path)
            self._logger.info("YOLO model loaded successfully.")
            return True
        except Exception as exc:
            self._logger.error("YOLO load failed: %s — falling back to DEMO mode.", exc)
            self._demo_mode = True
            return True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run inference on a frame and return structured DetectionResult."""
        result = DetectionResult(annotated_frame=frame.copy())

        if self._demo_mode:
            return self._simulated_detection(result)

        try:
            yolo_results = self._model(frame, verbose=False)[0]
            return self._parse_results(yolo_results, result)
        except Exception as exc:
            self._logger.error("Inference error: %s", exc)
            return result

    def _parse_results(self, yolo_results, result: DetectionResult) -> DetectionResult:
        annotated = result.annotated_frame
        h, w = annotated.shape[:2]

        for box in yolo_results.boxes:
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = yolo_results.names[cls].lower()

            if conf < self.CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_survivor = label in self.SURVIVOR_CLASSES
            is_obstacle = label in self.OBSTACLE_CLASSES

            if is_survivor:
                result.survivor_detected = True
                color = self.COLOR_SURVIVOR
                tag   = "SURVIVOR"
            elif is_obstacle:
                result.obstacle_detected = True
                color = self.COLOR_OBSTACLE
                tag   = "OBSTACLE"
            else:
                color = self.COLOR_OTHER
                tag   = label.upper()

            # ── Draw bounding box with thick border + filled header
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            header_h = 22
            cv2.rectangle(annotated, (x1, y1 - header_h), (x2, y1), color, -1)
            text = f"{tag} {conf:.0%}"
            cv2.putText(annotated, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 2, cv2.LINE_AA)

            result.detections.append({
                "label": label, "tag": tag,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        result.annotated_frame = annotated
        return result

    def _simulated_detection(self, result: DetectionResult) -> DetectionResult:
        """Randomized detections for demo / no-YOLO mode."""
        r = random.random()
        annotated = result.annotated_frame
        h, w = annotated.shape[:2]

        if r < 0.15:       # 15% chance survivor
            result.survivor_detected = True
            x1, y1, x2, y2 = w//3, h//4, 2*w//3, 3*h//4
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_SURVIVOR, 2)
            cv2.rectangle(annotated, (x1, y1-22), (x2, y1), self.COLOR_SURVIVOR, -1)
            cv2.putText(annotated, "SURVIVOR [DEMO] 87%", (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10,10,10), 2)
            result.detections.append({"tag": "SURVIVOR", "confidence": 0.87, "label": "person"})

        elif r < 0.35:     # 20% chance obstacle
            result.obstacle_detected = True
            x1, y1, x2, y2 = w//5, h//3, 2*w//5, 2*h//3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_OBSTACLE, 2)
            cv2.rectangle(annotated, (x1, y1-22), (x2, y1), self.COLOR_OBSTACLE, -1)
            cv2.putText(annotated, "OBSTACLE [DEMO] 74%", (x1+4, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10,10,10), 2)
            result.detections.append({"tag": "OBSTACLE", "confidence": 0.74, "label": "chair"})

        result.annotated_frame = annotated
        return result


# =============================================================================
# MODULE 3 — SENSOR FUSION MODULE
# =============================================================================

class SensorFusionModule:
    """
    Fuses visual detections with simulated LiDAR distance data.
    In a real rover this would receive actual sensor bus readings.
    """

    OBSTACLE_DISTANCE_THRESHOLD_CM = 40.0

    # LiDAR simulation parameters
    LIDAR_MIN_CM  = 10.0
    LIDAR_MAX_CM  = 300.0
    LIDAR_NOISE   = 5.0        # ± cm random noise

    def __init__(self):
        self._logger = logging.getLogger("SensorFusionModule")
        self._smoothed_distance = 150.0    # Running average for smooth display
        self._alpha = 0.25                 # EMA smoothing factor

    def update(self, detection: DetectionResult) -> SensorData:
        """Simulate LiDAR and fuse with visual detections."""
        raw_lidar = self._simulate_lidar(detection.obstacle_detected)

        # Exponential moving average for realistic sensor smoothing
        self._smoothed_distance = (
            self._alpha * raw_lidar + (1 - self._alpha) * self._smoothed_distance
        )

        obstacle_confirmed = (
            detection.obstacle_detected
            and self._smoothed_distance < self.OBSTACLE_DISTANCE_THRESHOLD_CM
        )

        data = SensorData(
            lidar_distance_cm  = round(self._smoothed_distance, 1),
            obstacle_confirmed = obstacle_confirmed,
            timestamp          = time.time()
        )

        if obstacle_confirmed:
            self._logger.info(
                "OBSTACLE CONFIRMED — LiDAR: %.1f cm (threshold: %.0f cm)",
                data.lidar_distance_cm, self.OBSTACLE_DISTANCE_THRESHOLD_CM
            )

        return data

    def _simulate_lidar(self, obstacle_nearby: bool) -> float:
        """
        When an obstacle is visually detected, bias LiDAR toward close range
        to simulate realistic sensor agreement.
        """
        if obstacle_nearby:
            # Biased close-range reading with noise
            base = random.uniform(self.LIDAR_MIN_CM, self.OBSTACLE_DISTANCE_THRESHOLD_CM + 20)
        else:
            # Clear path — far reading
            base = random.uniform(80.0, self.LIDAR_MAX_CM)

        return max(self.LIDAR_MIN_CM,
                   min(self.LIDAR_MAX_CM, base + random.gauss(0, self.LIDAR_NOISE)))


# =============================================================================
# MODULE 4 — DECISION ENGINE
# =============================================================================

class DecisionEngine:
    """
    Priority-based decision tree implementing autonomous rover logic.
    Higher priority conditions override lower ones.
    """

    def __init__(self):
        self._logger = logging.getLogger("DecisionEngine")
        self._last_action = ""

    def decide(self, state: SystemState) -> Tuple[str, RoverMovementState]:
        """
        Evaluate system state and return (action_string, movement_state).
        Priority order (highest first):
          1. Emergency stop override
          2. Communication lost
          3. Critical battery
          4. Survivor detected
          5. Obstacle confirmed
          6. Default forward motion
        """
        d = state.detection
        s = state.sensor_data

        if state.emergency_stop:
            action = "⚠ EMERGENCY STOP ACTIVATED"
            movement = RoverMovementState.EMERGENCY

        elif state.communication_lost:
            action = "✖ COMM LOST — HALTING ALL MOTION"
            movement = RoverMovementState.STOPPED

        elif state.battery_level < 15.0:
            action = f"⚡ LOW BATTERY ({state.battery_level:.0f}%) — RETURNING TO BASE"
            movement = RoverMovementState.RETURNING

        elif d.survivor_detected:
            action = "★ SURVIVOR DETECTED — STOPPED & MARKING LOCATION"
            movement = RoverMovementState.STOPPED

        elif s.obstacle_confirmed:
            action = "▲ OBSTACLE CONFIRMED — EXECUTING AVOIDANCE MANEUVER"
            movement = RoverMovementState.AVOIDING

        else:
            action = "▶ NOMINAL — MOVING FORWARD"
            movement = RoverMovementState.FORWARD

        # Log only on state transition
        if action != self._last_action:
            self._logger.info("Decision: %s | LiDAR: %.1f cm | Battery: %.1f%%",
                              action, s.lidar_distance_cm, state.battery_level)
            self._last_action = action

        return action, movement


# =============================================================================
# MODULE 5 — ROVER CONTROLLER
# =============================================================================

class RoverController:
    """
    Translates decisions into rover movement commands.
    In a real system this would drive motor controllers via serial/CAN bus.
    """

    def __init__(self):
        self._logger = logging.getLogger("RoverController")
        self._current_state = RoverMovementState.STOPPED
        self._avoidance_step = 0

    def execute(self, movement_state: RoverMovementState, state: SystemState):
        """Apply movement state with transition logging."""
        if movement_state != self._current_state:
            self._logger.info(
                "Movement transition: %s → %s",
                self._current_state.value, movement_state.value
            )
            self._current_state = movement_state
            self._avoidance_step = 0

        # Avoidance state machine (logged only)
        if movement_state == RoverMovementState.AVOIDING:
            self._avoidance_step += 1
            phase = self._avoidance_step % 90
            if phase < 30:
                self._logger.debug("Avoidance phase: REVERSE")
            elif phase < 60:
                self._logger.debug("Avoidance phase: TURN LEFT")
            else:
                self._logger.debug("Avoidance phase: ASSESS CLEAR PATH")

        state.movement_state = movement_state

    @property
    def current_state(self) -> RoverMovementState:
        return self._current_state


# =============================================================================
# MODULE 6 — DISPLAY / HUD SYSTEM
# =============================================================================

class DisplaySystem:
    """
    Renders the full robotics HUD overlay onto the annotated camera frame.
    Uses a scanline-style dark theme reminiscent of real field robotics UIs.
    """

    # ── Color palette (BGR)
    C_BG        = (10,  15,  20)
    C_PANEL     = (18,  25,  35)
    C_ACCENT    = (0,   200, 120)    # green accent
    C_WARN      = (0,   180, 255)    # amber-orange warning
    C_DANGER    = (40,  40,  230)    # red danger
    C_TEXT      = (210, 220, 225)
    C_DIM       = (90,  100, 110)
    C_SURVIVOR  = (0,   255, 130)
    C_OBSTACLE  = (30,  100, 255)
    C_MODE_EQ   = (80,  180, 255)    # earthquake amber
    C_MODE_FIRE = (30,  80,  220)    # fire red
    C_MODE_FLOOD= (200, 140, 40)     # flood blue

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO  = cv2.FONT_HERSHEY_PLAIN

    def __init__(self, display_w: int = 1280, display_h: int = 720):
        self.W = display_w
        self.H = display_h
        self._shake_offset = (0, 0)

    # ─────────────────────────────────────────────────────────────────────────
    def render(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Compose the complete HUD on top of the camera frame."""
        # Resize frame to display resolution
        frame = cv2.resize(frame, (self.W, self.H))

        # Apply disaster mode visual effect
        frame = self._apply_disaster_overlay(frame, state.disaster_mode)

        # Apply screen shake for earthquake mode
        frame = self._apply_shake(frame, state.disaster_mode, state.frame_count)

        # Render HUD panels
        frame = self._draw_top_bar(frame, state)
        frame = self._draw_left_panel(frame, state)
        frame = self._draw_right_panel(frame, state)
        frame = self._draw_action_banner(frame, state)
        frame = self._draw_scanlines(frame)
        frame = self._draw_crosshair(frame, state)
        frame = self._draw_corner_brackets(frame)

        return frame

    # ── Disaster visual effects ───────────────────────────────────────────────

    def _apply_disaster_overlay(self, frame: np.ndarray, mode: DisasterMode) -> np.ndarray:
        overlay = frame.copy()
        if mode == DisasterMode.FIRE:
            red_layer = np.zeros_like(frame)
            red_layer[:, :, 2] = 80   # boost red channel
            cv2.addWeighted(frame, 0.75, red_layer, 0.25, 0, overlay)
            # Vignette flicker
            alpha = 0.15 + 0.10 * abs(np.sin(time.time() * 8))
            vignette = self._make_vignette(frame.shape, (0, 40, 180))
            cv2.addWeighted(overlay, 1 - alpha, vignette, alpha, 0, overlay)

        elif mode == DisasterMode.FLOOD:
            blue_layer = np.zeros_like(frame)
            blue_layer[:, :, 0] = 90  # boost blue channel
            cv2.addWeighted(frame, 0.78, blue_layer, 0.22, 0, overlay)
            # Ripple scan line
            t = time.time()
            y_line = int((np.sin(t * 2) * 0.5 + 0.5) * frame.shape[0])
            cv2.line(overlay, (0, y_line), (frame.shape[1], y_line), (255, 200, 100), 1)

        elif mode == DisasterMode.EARTHQUAKE:
            # Desaturate slightly + noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(frame, 0.70, gray_3ch, 0.30, 0, overlay)
            noise = np.random.randint(0, 20, frame.shape, dtype=np.uint8)
            overlay = cv2.add(overlay, noise)

        return overlay

    def _apply_shake(self, frame: np.ndarray, mode: DisasterMode, frame_count: int) -> np.ndarray:
        if mode != DisasterMode.EARTHQUAKE:
            return frame
        magnitude = 6
        dx = int(random.gauss(0, magnitude))
        dy = int(random.gauss(0, magnitude))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    def _make_vignette(self, shape, color_bgr) -> np.ndarray:
        h, w = shape[:2]
        vignette = np.zeros((h, w, 3), dtype=np.uint8)
        for i, c in enumerate(color_bgr):
            kernel_x = cv2.getGaussianKernel(w, w // 2)
            kernel_y = cv2.getGaussianKernel(h, h // 2)
            kernel   = kernel_y * kernel_x.T
            mask     = (kernel / kernel.max() * 255).astype(np.uint8)
            mask_inv = 255 - mask
            vignette[:, :, i] = mask_inv
        return vignette

    # ── HUD panels ────────────────────────────────────────────────────────────

    def _draw_top_bar(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Top bar: title, mode, FPS, session time."""
        bar_h = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, bar_h), self.C_BG, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # Title
        cv2.putText(frame, "AUTONOMOUS DISASTER RESPONSE ROVER  v2.1",
                    (14, 30), self.FONT, 0.62, self.C_ACCENT, 1, cv2.LINE_AA)

        # Mode badge
        mode = state.disaster_mode
        mode_color = {
            DisasterMode.NONE:       self.C_DIM,
            DisasterMode.EARTHQUAKE: self.C_MODE_EQ,
            DisasterMode.FIRE:       self.C_MODE_FIRE,
            DisasterMode.FLOOD:      self.C_MODE_FLOOD,
        }[mode]
        badge_text = f"[ {mode.value} MODE ]"
        badge_x = self.W // 2 - 100
        cv2.putText(frame, badge_text, (badge_x, 30),
                    self.FONT, 0.64, mode_color, 2, cv2.LINE_AA)

        # FPS & uptime
        elapsed = int(time.time() - state.session_start)
        uptime  = f"{elapsed//60:02d}:{elapsed%60:02d}"
        fps_str = f"FPS: {state.fps:.0f}  |  UP: {uptime}"
        cv2.putText(frame, fps_str, (self.W - 200, 30),
                    self.FONT, 0.52, self.C_DIM, 1, cv2.LINE_AA)

        # Separator line
        cv2.line(frame, (0, bar_h), (self.W, bar_h), self.C_ACCENT, 1)
        return frame

    def _draw_left_panel(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Left panel: battery, comms, movement state."""
        px, py = 14, 70
        pw, ph = 220, 280
        self._panel_bg(frame, px, py, pw, ph)

        cy = py + 26
        self._section_header(frame, "POWER & COMMS", px + 8, cy)
        cy += 32

        # Battery bar
        bat = state.battery_level
        bat_color = (
            self.C_DANGER if bat < 15
            else self.C_WARN if bat < 40
            else self.C_ACCENT
        )
        cv2.putText(frame, f"BATTERY", (px + 8, cy),
                    self.FONT, 0.48, self.C_DIM, 1, cv2.LINE_AA)
        cy += 16
        bar_w = pw - 20
        filled = int(bar_w * bat / 100)
        cv2.rectangle(frame, (px+8, cy), (px+8+bar_w, cy+14), (40,50,60), -1)
        cv2.rectangle(frame, (px+8, cy), (px+8+filled, cy+14), bat_color, -1)
        cv2.rectangle(frame, (px+8, cy), (px+8+bar_w, cy+14), self.C_DIM, 1)
        cv2.putText(frame, f"{bat:.1f}%", (px + pw//2 - 18, cy + 11),
                    self.FONT, 0.40, (10,10,10), 1, cv2.LINE_AA)
        cy += 28

        # Comm status
        comm_color = self.C_DANGER if state.communication_lost else self.C_ACCENT
        comm_text  = "✖ LOST" if state.communication_lost else "✔ CONNECTED"
        cv2.putText(frame, "COMMS:", (px+8, cy), self.FONT, 0.48, self.C_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, comm_text, (px+80, cy), self.FONT, 0.48, comm_color, 1, cv2.LINE_AA)
        cy += 26

        # E-stop
        estop_color = self.C_DANGER if state.emergency_stop else self.C_DIM
        estop_text  = "ACTIVE" if state.emergency_stop else "STANDBY"
        cv2.putText(frame, "E-STOP:", (px+8, cy), self.FONT, 0.48, self.C_DIM, 1, cv2.LINE_AA)
        cv2.putText(frame, estop_text, (px+80, cy), self.FONT, 0.48, estop_color, 1, cv2.LINE_AA)
        cy += 32

        self._section_header(frame, "MOVEMENT STATE", px + 8, cy)
        cy += 24
        mv_colors = {
            RoverMovementState.FORWARD:   self.C_ACCENT,
            RoverMovementState.STOPPED:   self.C_WARN,
            RoverMovementState.AVOIDING:  self.C_WARN,
            RoverMovementState.RETURNING: self.C_MODE_EQ,
            RoverMovementState.EMERGENCY: self.C_DANGER,
        }
        mv_color = mv_colors.get(state.movement_state, self.C_TEXT)
        mv_lines = state.movement_state.value.split(" ")
        for ln in mv_lines:
            cv2.putText(frame, ln, (px+8, cy), self.FONT, 0.50, mv_color, 1, cv2.LINE_AA)
            cy += 20

        return frame

    def _draw_right_panel(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Right panel: detection status and LiDAR."""
        pw, ph = 230, 280
        px = self.W - pw - 14
        py = 70
        self._panel_bg(frame, px, py, pw, ph)

        cy = py + 26
        self._section_header(frame, "PERCEPTION", px + 8, cy)
        cy += 32

        # Survivor
        surv = state.detection.survivor_detected
        surv_color = self.C_SURVIVOR if surv else self.C_DIM
        surv_text  = "★ DETECTED" if surv else "○ NOT DETECTED"
        cv2.putText(frame, "SURVIVOR:", (px+8, cy), self.FONT, 0.48, self.C_DIM, 1, cv2.LINE_AA)
        cy += 18
        cv2.putText(frame, surv_text, (px+8, cy), self.FONT, 0.50, surv_color, 1, cv2.LINE_AA)
        cy += 26

        # Obstacle
        obs = state.detection.obstacle_detected
        obs_conf = state.sensor_data.obstacle_confirmed
        obs_color = self.C_OBSTACLE if obs_conf else (self.C_WARN if obs else self.C_DIM)
        obs_text  = (
            "▲ CONFIRMED" if obs_conf
            else ("△ VISUAL ONLY" if obs else "○ CLEAR")
        )
        cv2.putText(frame, "OBSTACLE:", (px+8, cy), self.FONT, 0.48, self.C_DIM, 1, cv2.LINE_AA)
        cy += 18
        cv2.putText(frame, obs_text, (px+8, cy), self.FONT, 0.50, obs_color, 1, cv2.LINE_AA)
        cy += 32

        self._section_header(frame, "LiDAR SENSOR", px + 8, cy)
        cy += 26

        dist = state.sensor_data.lidar_distance_cm
        dist_color = (
            self.C_DANGER if dist < 40
            else self.C_WARN if dist < 80
            else self.C_ACCENT
        )
        cv2.putText(frame, f"{dist:.1f} cm", (px+8, cy),
                    self.FONT, 0.80, dist_color, 2, cv2.LINE_AA)
        cy += 22
        cv2.putText(frame, "(simulated)", (px+8, cy), self.FONT, 0.38, self.C_DIM, 1)
        cy += 26

        # LiDAR bar
        bar_w  = pw - 20
        max_cm = 300.0
        filled = int(bar_w * min(dist, max_cm) / max_cm)
        cv2.rectangle(frame, (px+8, cy), (px+8+bar_w, cy+10), (40,50,60), -1)
        cv2.rectangle(frame, (px+8, cy), (px+8+filled, cy+10), dist_color, -1)
        cv2.rectangle(frame, (px+8, cy), (px+8+bar_w, cy+10), self.C_DIM, 1)
        # Threshold marker
        thresh_x = px + 8 + int(bar_w * 40 / max_cm)
        cv2.line(frame, (thresh_x, cy-2), (thresh_x, cy+12), (100,100,255), 2)

        return frame

    def _draw_action_banner(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Bottom action banner showing current rover decision."""
        bh = 46
        by = self.H - bh
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, by), (self.W, self.H), self.C_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, by), (self.W, by), self.C_ACCENT, 1)

        action = state.current_action
        action_color = (
            self.C_DANGER if "EMERGENCY" in action or "STOP" in action
            else self.C_WARN if "OBSTACLE" in action or "BATTERY" in action
            else self.C_SURVIVOR if "SURVIVOR" in action
            else self.C_ACCENT
        )

        # Pulsing dot
        pulse = int((np.sin(time.time() * 4) * 0.5 + 0.5) * 255)
        dot_color = (0, pulse, pulse // 2)
        cv2.circle(frame, (22, by + 22), 7, dot_color, -1)
        cv2.circle(frame, (22, by + 22), 7, self.C_DIM, 1)

        cv2.putText(frame, f"ROVER ACTION:  {action}",
                    (40, by + 29), self.FONT, 0.68, action_color, 2, cv2.LINE_AA)

        # Key hints
        hints = "[1] EQ  [2] FIRE  [3] FLOOD  [C] TOGGLE COMM  [E] EMERGENCY STOP  [Q] QUIT"
        cv2.putText(frame, hints, (40, by + 43),
                    self.FONT, 0.34, self.C_DIM, 1, cv2.LINE_AA)

        return frame

    def _draw_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Subtle scanline effect for CRT/military terminal aesthetics."""
        overlay = frame.copy()
        for y in range(0, self.H, 4):
            cv2.line(overlay, (0, y), (self.W, y), (0, 0, 0), 1)
        cv2.addWeighted(frame, 0.92, overlay, 0.08, 0, frame)
        return frame

    def _draw_crosshair(self, frame: np.ndarray, state: SystemState) -> np.ndarray:
        """Center crosshair / targeting reticle."""
        cx, cy = self.W // 2, self.H // 2
        size   = 22
        gap    = 7
        color  = self.C_ACCENT if not state.detection.survivor_detected else self.C_SURVIVOR
        thick  = 1

        # Horizontal
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, thick)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, thick)
        # Vertical
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, thick)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, thick)
        # Center dot
        cv2.circle(frame, (cx, cy), 2, color, -1)
        return frame

    def _draw_corner_brackets(self, frame: np.ndarray) -> np.ndarray:
        """Military-style corner brackets for framing."""
        L = 30
        T = 2
        c = self.C_ACCENT
        pad = 10
        # Top-left
        cv2.line(frame, (pad, pad), (pad+L, pad), c, T)
        cv2.line(frame, (pad, pad), (pad, pad+L), c, T)
        # Top-right
        cv2.line(frame, (self.W-pad, pad), (self.W-pad-L, pad), c, T)
        cv2.line(frame, (self.W-pad, pad), (self.W-pad, pad+L), c, T)
        # Bottom-left
        cv2.line(frame, (pad, self.H-pad), (pad+L, self.H-pad), c, T)
        cv2.line(frame, (pad, self.H-pad), (pad, self.H-pad-L), c, T)
        # Bottom-right
        cv2.line(frame, (self.W-pad, self.H-pad), (self.W-pad-L, self.H-pad), c, T)
        cv2.line(frame, (self.W-pad, self.H-pad), (self.W-pad, self.H-pad-L), c, T)
        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _panel_bg(self, frame, x, y, w, h, alpha=0.80):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), self.C_PANEL, -1)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), self.C_ACCENT, 1)
        cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0, frame)

    def _section_header(self, frame, text, x, y):
        cv2.line(frame, (x, y+2), (x+170, y+2), self.C_ACCENT, 1)
        cv2.putText(frame, text, (x, y), self.FONT, 0.43, self.C_ACCENT, 1, cv2.LINE_AA)


# =============================================================================
# MAIN APPLICATION — ROVER SYSTEM
# =============================================================================

class RoverSystem:
    """
    Orchestrates all modules through the main perception-decision-control loop.

    Input → CameraModule
    Perception → DetectionModule + SensorFusionModule
    Decision → DecisionEngine
    Control → RoverController
    Display → DisplaySystem
    """

    WINDOW_NAME = "AUTONOMOUS DISASTER RESPONSE ROVER  —  LIVE FEED"

    def __init__(self):
        self._logger = logging.getLogger("RoverSystem")

        # ── Instantiate all modules
        self.state      = SystemState(session_start=time.time())
        self.camera     = CameraModule(camera_index=0)
        self.detector   = DetectionModule(model_path="yolov8n.pt")
        self.fusion     = SensorFusionModule()
        self.decision   = DecisionEngine()
        self.controller = RoverController()
        self.display    = DisplaySystem(display_w=1280, display_h=720)

    def initialize(self) -> bool:
        self._logger.info("=" * 60)
        self._logger.info("DISASTER ROVER SYSTEM  —  BOOT SEQUENCE")
        self._logger.info("=" * 60)

        if not self.camera.initialize():
            self._logger.critical("Camera initialization failed. Aborting.")
            return False

        if not self.detector.initialize():
            self._logger.error("Detection module failed to load.")

        self._logger.info("All modules initialized. Starting main loop.")
        return True

    def run(self):
        """Main perception → decision → control loop."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1280, 720)

        while True:
            # ── 1. UPDATE SYSTEM STATE (battery, FPS)
            self.state.tick()

            # ── 2. INPUT — Capture frame
            ret, raw_frame = self.camera.read_frame()
            if not ret:
                self._logger.error("Frame capture lost. Retrying...")
                time.sleep(0.05)
                continue

            # ── 3. PERCEPTION — Detection
            yolo_frame = self.camera.get_yolo_frame(raw_frame)
            detection  = self.detector.detect(yolo_frame)

            # Scale annotated YOLO frame back to display size for overlay
            annotated = cv2.resize(detection.annotated_frame, (1280, 720))
            detection.annotated_frame = annotated

            self.state.detection = detection

            # ── 4. PERCEPTION — Sensor Fusion (LiDAR + vision)
            sensor_data = self.fusion.update(detection)
            self.state.sensor_data = sensor_data

            # ── 5. DECISION ENGINE
            action, movement = self.decision.decide(self.state)
            self.state.current_action = action

            # ── 6. CONTROL — Apply movement command
            self.controller.execute(movement, self.state)

            # ── 7. DISPLAY — Render HUD
            hud_frame = self.display.render(annotated, self.state)
            cv2.imshow(self.WINDOW_NAME, hud_frame)

            # ── 8. KEY HANDLING
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_keypress(key):
                break

        self._shutdown()

    def _handle_keypress(self, key: int) -> bool:
        """
        Returns False to signal shutdown.
        Key bindings:
          Q / ESC → quit
          1       → earthquake mode
          2       → fire mode
          3       → flood mode
          0       → standby mode
          C       → toggle comm loss
          E       → toggle emergency stop
        """
        if key in (ord('q'), ord('Q'), 27):   # ESC
            self._logger.info("Shutdown requested by operator.")
            return False

        elif key == ord('1'):
            self.state.disaster_mode = DisasterMode.EARTHQUAKE
            self._logger.info("DISASTER MODE: EARTHQUAKE")

        elif key == ord('2'):
            self.state.disaster_mode = DisasterMode.FIRE
            self._logger.info("DISASTER MODE: FIRE")

        elif key == ord('3'):
            self.state.disaster_mode = DisasterMode.FLOOD
            self._logger.info("DISASTER MODE: FLOOD")

        elif key == ord('0'):
            self.state.disaster_mode = DisasterMode.NONE
            self._logger.info("DISASTER MODE: STANDBY")

        elif key in (ord('c'), ord('C')):
            self.state.communication_lost = not self.state.communication_lost
            status = "LOST" if self.state.communication_lost else "RESTORED"
            self._logger.warning("FAILSAFE: Communication %s", status)

        elif key in (ord('e'), ord('E')):
            self.state.emergency_stop = not self.state.emergency_stop
            status = "ACTIVATED" if self.state.emergency_stop else "CLEARED"
            self._logger.warning("FAILSAFE: Emergency stop %s", status)

        return True

    def _shutdown(self):
        self._logger.info("Shutting down all subsystems...")
        self.camera.release()
        cv2.destroyAllWindows()
        elapsed = int(time.time() - self.state.session_start)
        self._logger.info(
            "Session ended. Runtime: %02d:%02d | Frames: %d | Final battery: %.1f%%",
            elapsed // 60, elapsed % 60, self.state.frame_count, self.state.battery_level
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    rover = RoverSystem()
    if rover.initialize():
        rover.run()
    else:
        print("\n[ERROR] System initialization failed. Check webcam and dependencies.\n")