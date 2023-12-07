import cv2
import mediapipe as mp 
import csv
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

inputWidth = 368
inputHeight = 368
inputScale = 1.0 / 255

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)



BODY_PARTS = {
    "Nose": 0, "LeftEye": 1, "RightEye": 2, "LeftEar": 3, "RightEar": 4,
    "LeftShoulder": 5, "RightShoulder": 6, "LeftElbow": 7, "RightElbow": 8,
    "LeftWrist": 9, "RightWrist": 10, "LeftHip": 11, "RightHip": 12,
    "LeftKnee": 13, "RightKnee": 14, "LeftAnkle": 15, "RightAnkle": 16
}

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

for connection in POSE_CONNECTIONS:
    part1, part2 = connection
    if part1 not in BODY_PARTS:
        BODY_PARTS[part1] = len(BODY_PARTS)
    if part2 not in BODY_PARTS:
        BODY_PARTS[part2] = len(BODY_PARTS)

def process_video_content(video_path):
    capture = cv2.VideoCapture(video_path)

    csv_filename = "joint_angle_video.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Frame', 'Joint', 'Angle (Degrees)']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        frame_number = 0

        while True:
            has_frame, frame = capture.read()
        
            if not has_frame:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(frame_rgb)
        
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                joint_data = {}

                for i, landmark in enumerate(landmarks):
                    x, y = int(landmark.x * inputWidth), int(landmark.y * inputHeight)
                    if landmark.visibility > 0.1:
                        joint_data[i] = (x, y)
                 
                angles = calculate_joint_angles(joint_data)
                for (part_from, part_to), angle_deg in angles.items():
                    csv_writer.writerow({'Frame': frame_number, 'Joint': f"{part_from} to {part_to}", 'Angle (Degrees)': angle_deg})

            frame_number += 1

        print(f"영상 관절 각도 CSV 저장: {csv_filename}")

        df = pd.read_csv(csv_filename)
        angles_data = df['Angle (Degrees)'].values.reshape(-1, 1)
        scaler.fit(angles_data)

        return csv_filename
    
def calculate_joint_angles(landmarks):
    angles = {}
    for pair in POSE_CONNECTIONS:
        part1, part2 = pair
        id1, id2 = BODY_PARTS[part1], BODY_PARTS[part2]

        if id1 in landmarks and id2 in landmarks:
            x1, y1 = landmarks[id1]
            x2, y2 = landmarks[id2]

            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            
            angles[(part1, part2)] = angle_deg

    return angles

process_video_content('KakaoTalk_20231204_025005558.mp4')