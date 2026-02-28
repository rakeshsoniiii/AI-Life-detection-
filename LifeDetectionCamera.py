import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover - informative runtime guard
    YOLO = None


# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("RescueRover")


# ============================================================
# ENUMS & DATA CLASSES
# ============================================================


class DisasterMode(Enum):
    EARTHQUAKE = "Earthquake"
    FIRE = "Fire"
    FLOOD = "Flood"


class RoverMotionState(Enum):
    FORWARD = "FORWARD"
    STOPPED = "STOPPED"
    AVOIDING = "AVOIDING"
    RETURNING = "RETURNING"


@dataclass
class SystemState:
    battery_level: float = 100.0
    communication_ok: bool = True
    emergency_stop: bool = False
    disaster_mode: DisasterMode = DisasterMode.EARTHQUAKE
    rover_motion_state: RoverMotionState = RoverMotionState.STOPPED

    survivor_detected: bool = False
    obstacle_detected: bool = False
    obstacle_confirmed: bool = False
    lidar_distance_cm: float = 0.0

    current_action: str = "INITIALIZING"
    running: bool = True

    last_action: Optional[str] = None
    last_log_time: float = field(default_factory=time.time)


@dataclass
class DetectionBox:
    xyxy: Tuple[int, int, int, int]
    label: str
    confidence: float
    color: Tuple[int, int, int]


@dataclass
class DetectionResult:
    survivor_detected: bool = False
    obstacle_detected: bool = False
    boxes: List[DetectionBox] = field(default_factory=list)


# ============================================================
# CAMERA MODULE
# ============================================================


class CameraModule:
    """
    Wraps the laptop webcam and provides frames for both display and YOLO.
    """

    def __init__(
        self,
        camera_index: int = 0,
        yolo_img_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.camera_index = camera_index
        self.yolo_img_size = yolo_img_size

        # On Windows, CAP_DSHOW is often more reliable for webcams.
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logger.error("Failed to open camera index %d", self.camera_index)
            self.cap = None
        else:
            logger.info("Camera opened on index %d", self.camera_index)

    def is_available(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (display_frame, yolo_frame).
        """
        if not self.is_available():
            return None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.error("Camera read failed - stopping system.")
            return None, None

        # YOLO will run on a resized copy to keep inference cheap.
        yolo_frame = cv2.resize(frame, self.yolo_img_size, interpolation=cv2.INTER_LINEAR)
        return frame, yolo_frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released.")


# ============================================================
# DETECTION MODULE (YOLOv8)
# ============================================================


class DetectionModule:
    """
    Runs YOLOv8 on incoming frames and classifies survivors/obstacles.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        if YOLO is None:
            logger.error(
                "Ultralytics YOLO is not installed. "
                "Install it with 'pip install ultralytics' to enable detection."
            )
            self.model = None
        else:
            logger.info("Loading YOLOv8 model: %s", model_path)
            self.model = YOLO(model_path)
            logger.info("YOLOv8 model loaded.")

        # COCO class names we treat as obstacles (simulated debris)
        self.obstacle_class_names = {
            "chair",
            "couch",
            "bed",
            "dining table",
            "bench",
            "backpack",
            "suitcase",
            "handbag",
            "car",
            "truck",
            "bus",
        }

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        result = DetectionResult()

        if self.model is None:
            # Detection disabled, return empty result.
            return result

        try:
            # Run at lower verbosity to keep console clear for engineering logs.
            yolo_results = self.model(frame_bgr, imgsz=frame_bgr.shape[0], verbose=False)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("YOLO inference failed: %s", exc)
            return result

        if not yolo_results:
            return result

        yres = yolo_results[0]
        h, w = frame_bgr.shape[:2]

        for box in yres.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            name = str(self.model.names.get(cls_id, cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1 * w / w), int(y1 * h / h), int(x2 * w / w), int(y2 * h / h)

            label = f"{name} {conf:.2f}"
            color = (0, 255, 0)  # default green

            # Survivor logic: "person" class
            if name == "person":
                result.survivor_detected = True
                color = (0, 255, 0)  # green

            # Obstacle logic: any "debris-like" class
            if name in self.obstacle_class_names:
                result.obstacle_detected = True
                color = (0, 165, 255)  # orange

            result.boxes.append(
                DetectionBox(
                    xyxy=(x1, y1, x2, y2),
                    label=label,
                    confidence=conf,
                    color=color,
                )
            )

        return result


# ============================================================
# SENSOR FUSION MODULE
# ============================================================


class SensorFusionModule:
    """
    Simulates LiDAR distance and fuses it with vision-based obstacle detection.
    """

    def __init__(self) -> None:
        self.min_distance_cm = 10.0
        self.max_distance_cm = 300.0

    def update(self, state: SystemState, detection: DetectionResult) -> None:
        # Biased sampling: if an obstacle is visually detected, push distances closer.
        if detection.obstacle_detected:
            distance = random.uniform(self.min_distance_cm, 150.0)
        else:
            distance = random.uniform(80.0, self.max_distance_cm)

        state.lidar_distance_cm = distance
        state.obstacle_confirmed = detection.obstacle_detected and distance < 40.0

        logger.info(
            "Sensors | LIDAR: %.1f cm | VisionObstacle: %s | ObstacleConfirmed: %s",
            state.lidar_distance_cm,
            detection.obstacle_detected,
            state.obstacle_confirmed,
        )


# ============================================================
# DECISION ENGINE
# ============================================================


class DecisionEngine:
    """
    Implements the priority-based decision tree for rover actions.
    """

    def decide(self, state: SystemState) -> Tuple[str, RoverMotionState]:
        # Priority tree as specified.
        if state.emergency_stop:
            action = "EMERGENCY STOP (MANUAL OVERRIDE)"
            motion = RoverMotionState.STOPPED
        elif not state.communication_ok:
            action = "EMERGENCY STOP - COMMUNICATION LOST"
            motion = RoverMotionState.STOPPED
        elif state.battery_level < 15.0:
            action = "LOW BATTERY - RETURN TO BASE"
            motion = RoverMotionState.RETURNING
        elif state.survivor_detected:
            action = "SURVIVOR DETECTED - STOP AND MARK LOCATION"
            motion = RoverMotionState.STOPPED
        elif state.obstacle_confirmed:
            action = "OBSTACLE AHEAD - AVOIDING"
            motion = RoverMotionState.AVOIDING
        else:
            action = "MOVING FORWARD"
            motion = RoverMotionState.FORWARD

        state.current_action = action
        state.rover_motion_state = motion

        # Structured logging, but only on changes to reduce spam.
        now = time.time()
        if action != state.last_action or (now - state.last_log_time) > 2.0:
            logger.info(
                "Decision | Action: %s | MotionState: %s | Battery: %.1f%% | CommOK: %s",
                action,
                motion.value,
                state.battery_level,
                state.communication_ok,
            )
            state.last_action = action
            state.last_log_time = now

        return action, motion


# ============================================================
# ROVER CONTROLLER
# ============================================================


class RoverController:
    """
    High-level orchestrator for input → perception → decision → control → display.
    """

    def __init__(
        self,
        camera: CameraModule,
        detector: DetectionModule,
        fusion: SensorFusionModule,
        decision_engine: DecisionEngine,
        state: SystemState,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.fusion = fusion
        self.decision_engine = decision_engine
        self.state = state

        self._last_time = time.time()

    # ------------------------------
    # Simulation of internal rover state (movement, battery, comms)
    # ------------------------------

    def _update_battery(self) -> None:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # Base drain rate tuned for a few minutes of demo runtime.
        base_drain_per_sec = 0.08

        # Higher load when avoiding or returning.
        if self.state.rover_motion_state in (RoverMotionState.AVOIDING, RoverMotionState.RETURNING):
            base_drain_per_sec *= 1.5

        # Disaster-mode environmental effects.
        if self.state.disaster_mode == DisasterMode.FIRE:
            base_drain_per_sec *= 1.7  # overheating penalty

        drain = base_drain_per_sec * dt
        self.state.battery_level = max(0.0, self.state.battery_level - drain)

        if self.state.battery_level <= 0.0:
            self.state.emergency_stop = True
            logger.warning("Battery depleted - forcing emergency stop.")

    def _handle_key_input(self, key: int) -> None:
        if key == ord("q") or key == 27:  # ESC
            logger.info("Shutdown requested by operator.")
            self.state.running = False
        elif key == ord("c"):
            # Toggle communication link
            self.state.communication_ok = not self.state.communication_ok
            if not self.state.communication_ok:
                logger.warning("Communication link LOST (operator toggle).")
            else:
                logger.info("Communication link RESTORED (operator toggle).")
        elif key == ord("e"):
            # Emergency stop override toggle
            self.state.emergency_stop = not self.state.emergency_stop
            if self.state.emergency_stop:
                logger.warning("Emergency STOP engaged by operator.")
            else:
                logger.info("Emergency STOP cleared by operator.")
        elif key == ord("1"):
            self.state.disaster_mode = DisasterMode.EARTHQUAKE
            logger.info("Disaster mode set to EARTHQUAKE.")
        elif key == ord("2"):
            self.state.disaster_mode = DisasterMode.FIRE
            logger.info("Disaster mode set to FIRE.")
        elif key == ord("3"):
            self.state.disaster_mode = DisasterMode.FLOOD
            logger.info("Disaster mode set to FLOOD.")

    # ------------------------------
    # Visualization helpers
    # ------------------------------

    def _apply_disaster_visual_effects(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        if self.state.disaster_mode == DisasterMode.EARTHQUAKE:
            # Subtle random shake.
            max_shift = 4
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        elif self.state.disaster_mode == DisasterMode.FIRE:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 140, 255), -1)  # orange tint in BGR
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        elif self.state.disaster_mode == DisasterMode.FLOOD:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (255, 0, 0), -1)  # blue tint
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        return frame

    def _draw_detections(self, frame: np.ndarray, detection: DetectionResult) -> None:
        for box in detection.boxes:
            x1, y1, x2, y2 = box.xyxy
            cv2.rectangle(frame, (x1, y1), (x2, y2), box.color, 2)
            cv2.putText(
                frame,
                box.label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box.color,
                2,
                lineType=cv2.LINE_AA,
            )

    def _draw_hud(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]

        # Battery bar
        bar_x1, bar_y1 = 20, 20
        bar_x2, bar_y2 = 220, 40
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 1)

        level = max(0.0, min(100.0, self.state.battery_level))
        fill_width = int((bar_x2 - bar_x1 - 2) * (level / 100.0))
        if level > 50:
            color = (0, 255, 0)  # green
        elif level > 20:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 0, 255)  # red

        cv2.rectangle(
            frame,
            (bar_x1 + 1, bar_y1 + 1),
            (bar_x1 + 1 + fill_width, bar_y2 - 1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            f"{int(level)}%",
            (bar_x1 + 5, bar_y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # Communication status
        comm_text = "COMM: OK" if self.state.communication_ok else "COMM: LOST"
        comm_color = (0, 255, 0) if self.state.communication_ok else (0, 0, 255)
        cv2.putText(
            frame,
            comm_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            comm_color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Disaster mode
        cv2.putText(
            frame,
            f"MODE: {self.state.disaster_mode.value}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Current action
        cv2.putText(
            frame,
            f"ACTION: {self.state.current_action}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Obstacle and survivor status
        obstacle_text = f"OBSTACLE: {'YES' if self.state.obstacle_detected else 'NO'}"
        obstacle_color = (0, 0, 255) if self.state.obstacle_confirmed else (0, 255, 0)
        cv2.putText(
            frame,
            obstacle_text + (f" (CONFIRMED)" if self.state.obstacle_confirmed else ""),
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            obstacle_color,
            2,
            lineType=cv2.LINE_AA,
        )

        survivor_text = f"SURVIVOR: {'DETECTED' if self.state.survivor_detected else 'NONE'}"
        survivor_color = (0, 255, 0) if self.state.survivor_detected else (255, 255, 255)
        cv2.putText(
            frame,
            survivor_text,
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            survivor_color,
            2,
            lineType=cv2.LINE_AA,
        )

        # Rover motion state at bottom left
        cv2.putText(
            frame,
            f"MOTION STATE: {self.state.rover_motion_state.value}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

        # Small operator hint
        hint = "Keys: Q/Esc=Quit | C=Comm toggle | E=Emergency stop | 1=Earthquake 2=Fire 3=Flood"
        cv2.putText(
            frame,
            hint,
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA,
        )

    # ------------------------------
    # Main control loop
    # ------------------------------

    def run(self) -> None:
        if not self.camera.is_available():
            logger.error("Camera not available. Aborting simulation.")
            return

        logger.info("Starting Autonomous Disaster Response Rover simulation.")

        while self.state.running:
            display_frame, yolo_frame = self.camera.read()
            if display_frame is None or yolo_frame is None:
                logger.error("No frame received from camera. Stopping.")
                break

            # PERCEPTION: YOLO detection
            detection = self.detector.detect(yolo_frame)
            self.state.survivor_detected = detection.survivor_detected
            self.state.obstacle_detected = detection.obstacle_detected

            # SENSOR FUSION: LIDAR + detection
            self.fusion.update(self.state, detection)

            # DECISION: compute action and internal motion state
            self._update_battery()
            self.decision_engine.decide(self.state)

            # CONTROL: (conceptual for this prototype)
            # In a real rover, this is where wheel commands, steering, etc. would be issued
            # based on self.state.rover_motion_state.

            # DISPLAY: apply visual effects, draw detections and HUD.
            display_frame = self._apply_disaster_visual_effects(display_frame)
            # Map boxes from YOLO-resized space back to display frame proportions if sizes differ.
            if yolo_frame.shape[:2] != display_frame.shape[:2]:
                scale_x = display_frame.shape[1] / yolo_frame.shape[1]
                scale_y = display_frame.shape[0] / yolo_frame.shape[0]
                scaled_boxes = []
                for b in detection.boxes:
                    x1, y1, x2, y2 = b.xyxy
                    sx1 = int(x1 * scale_x)
                    sy1 = int(y1 * scale_y)
                    sx2 = int(x2 * scale_x)
                    sy2 = int(y2 * scale_y)
                    scaled_boxes.append(
                        DetectionBox(
                            xyxy=(sx1, sy1, sx2, sy2),
                            label=b.label,
                            confidence=b.confidence,
                            color=b.color,
                        )
                    )
                detection_to_draw = DetectionResult(
                    survivor_detected=detection.survivor_detected,
                    obstacle_detected=detection.obstacle_detected,
                    boxes=scaled_boxes,
                )
            else:
                detection_to_draw = detection

            self._draw_detections(display_frame, detection_to_draw)
            self._draw_hud(display_frame)

            cv2.imshow("AI RESCUE ROVER - CAMERA VIEW", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self._handle_key_input(key)

        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Simulation terminated.")


# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:
    state = SystemState()
    camera = CameraModule()
    detector = DetectionModule()
    fusion = SensorFusionModule()
    decision_engine = DecisionEngine()

    controller = RoverController(
        camera=camera,
        detector=detector,
        fusion=fusion,
        decision_engine=decision_engine,
        state=state,
    )
    controller.run()


if __name__ == "__main__":
    main()

