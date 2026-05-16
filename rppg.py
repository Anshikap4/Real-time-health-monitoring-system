import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

# ---------------- SETTINGS ---------------- #
FPS_TARGET = 30
BUFFER_SIZE = 150
MIN_HZ = 0.7
MAX_HZ = 4.0

# ---------------- CAMERA ---------------- #
def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Camera not accessible")
    return cap

# ---------------- HEART RATE ---------------- #
def estimate_hr(signal, fps):
    if len(signal) < 32:
        return None

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

    freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
    fft = np.abs(np.fft.rfft(signal))

    idx = np.where((freqs >= MIN_HZ) & (freqs <= MAX_HZ))[0]
    if len(idx) == 0:
        return None

    peak = idx[np.argmax(fft[idx])]
    return freqs[peak] * 60

# ---------------- EMOTION ---------------- #
def estimate_emotion(face):
    if face is None or face.size == 0:
        return "No Face"

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Simple but working heuristic
    brightness = np.mean(gray)
    variation = np.std(gray)

    if brightness > 150 and variation > 30:
        return "Happy 🙂"
    elif brightness < 80:
        return "Sad 😔"
    elif variation < 20:
        return "Calm 😌"
    else:
        return "Neutral 😐"

# ---------------- BLOOD PRESSURE ---------------- #
def estimate_bp(hr):
    if hr is None:
        return 120, 80
    sys = 100 + (hr * 0.4)
    dia = 65 + (hr * 0.25)
    return int(sys), int(dia)

# ---------------- MAIN ---------------- #
def main():
    print("RUNNING NEW FULL CODE ✅")

    cap = open_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    green_buffer = deque(maxlen=BUFFER_SIZE)
    time_buffer = deque(maxlen=BUFFER_SIZE)

    hr_smooth = None
    emotion = "Starting..."

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ih, iw = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        face_roi = None

        # -------- FACE DETECTION -------- #
        if results.detections:
            det = max(results.detections,
                      key=lambda d: d.location_data.relative_bounding_box.width *
                                    d.location_data.relative_bounding_box.height)

            bbox = det.location_data.relative_bounding_box

            x1 = max(0, int(bbox.xmin * iw))
            y1 = max(0, int(bbox.ymin * ih))
            x2 = min(iw, x1 + int(bbox.width * iw))
            y2 = min(ih, y1 + int(bbox.height * ih))

            face_roi = frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # -------- SIGNAL EXTRACTION -------- #
            if face_roi.size > 0:
                roi = face_roi[0:int(0.3 * face_roi.shape[0]), :]
                if roi.size > 0:
                    green_buffer.append(np.mean(roi[:, :, 1]))
                    time_buffer.append(time.time())

        # -------- EMOTION -------- #
        emotion = estimate_emotion(face_roi)

        # -------- FPS -------- #
        fps = FPS_TARGET
        if len(time_buffer) > 1:
            dt = np.mean(np.diff(time_buffer))
            if dt > 0:
                fps = 1.0 / dt

        # -------- HEART RATE -------- #
        hr_val = None
        if len(green_buffer) >= 32:
            hr_val = estimate_hr(np.array(green_buffer), fps)

            if hr_val is not None:
                if hr_smooth is None:
                    hr_smooth = hr_val
                else:
                    hr_smooth = 0.6 * hr_val + 0.4 * hr_smooth

        # -------- BLOOD PRESSURE -------- #
        sys, dia = estimate_bp(hr_smooth)

        # -------- TEXT -------- #
        hr_text = f"HR: {hr_smooth:.1f} BPM" if hr_smooth else "HR: Calculating..."
        emotion_text = f"Emotion: {emotion}"
        bp_text = f"BP: {sys}/{dia} mmHg"

        # -------- DISPLAY -------- #
        cv2.putText(frame, hr_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.putText(frame, emotion_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, bp_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # FPS display
        now = time.time()
        fps_disp = 1.0 / (now - last_time) if now != last_time else 0
        last_time = now

        cv2.putText(frame, f"FPS: {fps_disp:.1f}", (10, ih - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("AI Health Monitor (HR + Emotion + BP)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()