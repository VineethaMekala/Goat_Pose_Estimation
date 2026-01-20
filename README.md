# Goat_Pose_Estimation
# Goat Pose Estimation (YOLOv8 Pose) — Inference with Skeleton Lines (OpenCV)

This project performs **Goat Pose Estimation** using a trained **YOLOv8 Pose model (`best.pt`)** and visualizes:

==>keypoints  
==>keypoint index numbers  
==>skeleton connections (lines between keypoints)  
==>adjustable output window size  

The output is displayed as a video window using **OpenCV (`cv2.imshow`)**.

---

## 1) Requirements

### Software
- Python 3.8+ (Recommended: Python 3.10 / 3.11)
- VS Code (or any Python IDE)

### Python Packages
- `ultralytics`
- `opencv-python`

Install packages:

```bash
pip install ultralytics opencv-python
```


**## 2) Keypoints Used in Training (Order)**

The model was trained with these keypoints in the given order:

| Index | Keypoint Name |
| ----: | ------------- |
|     0 | nose          |
|     1 | left_ear      |
|     2 | right_ear     |
|     3 | neck          |
|     4 | front_knee1   |
|     5 | hip           |
|     6 | back_knee1    |
|     7 | back_knee2    |
|     8 | tail          |
|     9 | front_knee2   |
|    10 | front_hoof1   |
|    11 | front_hoof2   |
|    12 | back_hoof1    |
|    13 | back_hoof2    |
Important:
This keypoint order MUST match how the dataset was labeled during training.

**##3) Skeleton Connections (Lines)**

The skeleton is drawn by connecting keypoints using these pairs:

nose → left_ear, right_ear

left_ear → neck

right_ear → neck

neck → hip

hip → tail

neck → front_knee1

neck → front_knee2

front_knee1 → front_hoof1

front_knee2 → front_hoof2

hip → back_knee1

hip → back_knee2

back_knee1 → back_hoof1

back_knee2 → back_hoof2

The code defines skeleton as:

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


**## 4) How to Run Inference (Testing)**
**## Run using VS Code Terminal**

Open VS Code terminal and go to project folder:

cd path/to/GoatPoseRun
python Testing_code.py

Output:
A window will open showing:
keypoints
index numbers for each keypoint
skeleton lines between keypoints
Press q to quit.

**## 5) Testing Script (Testing_code.py)**

The script:

Loads model (Goat_Pose_best.pt)

Opens input video (Goat_Video.mp4)

Runs pose inference on each frame

Draws keypoints + numbers

Draws skeleton lines using OpenCV

Shows output window

**## 6) Adjust Output Window Size**

Inside the script you can change:
OUTPUT_W, OUTPUT_H = 1000, 650
