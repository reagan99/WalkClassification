import math
import csv
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 미디어파이프에 정의된 관절 이름 및 관계 사용
BODY_PARTS = {
    "Nose": 0, "LeftEye": 1, "RightEye": 2, "LeftEar": 3, "RightEar": 4,
    "LeftShoulder": 5, "RightShoulder": 6, "LeftElbow": 7, "RightElbow": 8,
    "LeftWrist": 9, "RightWrist": 10, "LeftHip": 11, "RightHip": 12,
    "LeftKnee": 13, "RightKnee": 14, "LeftAnkle": 15, "RightAnkle": 16
}

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# BODY_PARTS 딕셔너리 업데이트
for connection in POSE_CONNECTIONS:
    part1, part2 = connection
    if part1 not in BODY_PARTS:
        BODY_PARTS[part1] = len(BODY_PARTS)
    if part2 not in BODY_PARTS:
        BODY_PARTS[part2] = len(BODY_PARTS)
        
def calculate_joint_angles(landmarks):
    angles = {}

    # Define the BODY_PARTS and POSE_CONNECTIONS dictionaries here

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

def read_joint_data(csv_filename):
    joint_data = {}

    with open(csv_filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            frame = int(row[0])
            joint_index = int(row[1])
            x = float(row[2])
            y = float(row[3])

            if frame not in joint_data:
                joint_data[frame] = {}
            joint_data[frame][joint_index] = (x, y)

    return joint_data

def write_angle_data_to_csv(angle_data, output_csv_file):
    with open(output_csv_file, mode='w', newline='') as csv_file:
        fieldnames = ['Frame', 'Joint', 'Angle (Degrees)']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for frame, angles in angle_data.items():
            for (part_from, part_to), angle_deg in angles.items():
                csv_writer.writerow({'Frame': frame, 'Joint': f"{part_from} to {part_to}", 'Angle (Degrees)': angle_deg})

# CSV 파일 경로

# List of CSV files
csv_files = [
    "hs_1_mediapipe.csv",
    "hs_2_mediapipe.csv",
    "hs_3_mediapipe.csv",
    "hs_4_mediapipe.csv",
    "hs_5_mediapipe.csv",
    "hs_6_mediapipe.csv",
    "hs_7_mediapipe.csv",
    "hs_8_mediapipe.csv",
    "hs_9_mediapipe.csv",
    "hs_10_mediapipe.csv",
    "js_1_mediapipe.csv",
    "js_2_mediapipe.csv",
    "js_3_mediapipe.csv",
    "js_4_mediapipe.csv",
    "js_5_mediapipe.csv",
    "js_6_mediapipe.csv",
    "js_7_mediapipe.csv",
    "js_8_mediapipe.csv",
    "js_9_mediapipe.csv",
]

# Loop through each CSV file
for csv_file in csv_files:
    # CSV 파일에서 관절 데이터 읽어오기
    joint_data = read_joint_data(csv_file)

    # 관절 각도 계산 및 저장할 딕셔너리 초기화
    angle_data = {}

    # 관절 각도 계산 및 딕셔너리에 저장 (모든 프레임에 대해)
    for frame, landmarks in joint_data.items():
        angles = calculate_joint_angles(landmarks)
        angle_data[frame] = angles

    # 관절 각도 데이터를 CSV 파일로 저장
    output_csv_file = os.path.splitext(csv_file)[0] + "_angle.csv"
    write_angle_data_to_csv(angle_data, output_csv_file)

    print(f"관절 각도 데이터를 {output_csv_file} 파일로 저장했습니다.")
write_angle_data_to_csv(angle_data, output_csv_file)

print(f"관절 각도 데이터를 {output_csv_file} 파일로 저장했습니다.")
