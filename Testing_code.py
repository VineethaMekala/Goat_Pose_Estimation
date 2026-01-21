from ultralytics import YOLO
import cv2

# ---------------------------
# 1) Load your trained model
# ---------------------------
model = YOLO("Goat_Pose_best.pt")

# ---------------------------
# 2) Open the video file
# ---------------------------
video_path = "Goat_Video1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video file: {video_path}")

# ---------------------------
# 3) Define skeleton connections
# ---------------------------
skeleton = [
    (0, 1),   # nose → left_ear
    (0, 2),   # nose → right_ear

    (1, 3),   # left_ear → neck
    (2, 3),   # right_ear → neck

    (3, 5),   # neck → hip
    (5, 8),   # hip → tail

    (3, 4),   # neck → front_knee1
    (3, 9),   # neck → front_knee2

    (4, 10),  # front_knee1 → front_hoof1
    (9, 11),  # front_knee2 → front_hoof2

    (5, 6),   # hip → back_knee1
    (5, 7),   # hip → back_knee2

    (6, 12),  # back_knee1 → back_hoof1
    (7, 13)   # back_knee2 → back_hoof2
]

# ---------------------------
# 4) Keypoint text style
# ---------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ---------------------------
# 5) Output window settings
# ---------------------------
window_name = "Goat Pose Estimation with Skeleton + Index"

# Set desired output window size here
OUTPUT_W, OUTPUT_H = 1000, 650

# Create resizable window and set initial size
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, OUTPUT_W, OUTPUT_H)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------
    # 6) Run pose estimation
    # ---------------------------
    results = model(frame)

    # ---------------------------
    # 7) Draw base keypoints output
    # ---------------------------
    annotated = results[0].plot()

    # ---------------------------
    # 8) Extract keypoints and draw numbers + skeleton
    # ---------------------------
    if results[0].keypoints is not None:
        kpts = results[0].keypoints.xy  # (num_objects, num_keypoints, 2)

        for obj_kpts in kpts:
            # Draw keypoint numbers
            for idx, (x, y) in enumerate(obj_kpts):
                x, y = int(x), int(y)

                # Draw dot
                cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)

                # Draw keypoint index number
                cv2.putText(
                    annotated,
                    str(idx),
                    (x + 5, y - 5),
                    FONT,
                    FONT_SCALE,
                    (255, 255, 255),
                    FONT_THICKNESS,
                    cv2.LINE_AA
                )

            # Draw skeleton lines
            for p1, p2 in skeleton:
                x1, y1 = obj_kpts[p1]
                x2, y2 = obj_kpts[p2]

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---------------------------
    # 9) Resize output frame to match desired window size
    # ---------------------------
    annotated_resized = cv2.resize(annotated, (OUTPUT_W, OUTPUT_H))

    # ---------------------------
    # 10) Show output
    # ---------------------------
    cv2.imshow(window_name, annotated_resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

