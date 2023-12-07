import cv2
import mediapipe as mp
import os
import csv

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# MediaPipe Pose 모델 로드
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 비디오 파일 경로 설정
video_files = [
    "hs_1.mp4",
    "hs_2.mp4",
    "hs_3.mp4",
    "hs_4.mp4",
    "hs_5.mp4",
    "hs_6.mp4",
    "hs_7.mp4",
    "hs_8.mp4",
    "hs_9.mp4",
    "hs_10.mp4",
    "js_1.mp4",
    "js_2.mp4",
    "js_3.mp4",
    "js_4.mp4",
    "js_5.mp4",
    "js_6.mp4",
    "js_7.mp4",
    "js_8.mp4",
    "js_9.mp4",

]
inputWidth = 368
inputHeight = 368
inputScale = 1.0 / 255

for video_file in video_files:
    # 비디오 파일 열기
    capture = cv2.VideoCapture(video_file)

    # 출력 영상 파일 설정
    file_name = os.path.splitext(os.path.basename(video_file))[0]
    output_filename = f"{file_name}_mediapipe.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30
    output_width = int(capture.get(3))
    output_height = int(capture.get(4))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (output_width, output_height))

    # CSV 파일 설정
    csv_filename = f"{file_name}_mediapipe.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Frame', 'Joint', 'X', 'Y']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    frame_number = 0

    while True:
        hasFrame, frame = capture.read()

        if not hasFrame:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            for i, landmark in enumerate(landmarks):
                x, y = int(landmark.x * inputWidth), int(landmark.y * inputHeight)
                if landmark.visibility > 0.1:
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        csv_writer.writerow({'Frame': frame_number, 'Joint': i, 'X': x, 'Y': y})

        out.write(frame)
        frame_number += 1

    # 파일 닫기
    capture.release()
    out.release()

    print(f"MediaPipe로 관절 추적, 스켈레톤 그리기 및 CSV 저장 완료: {output_filename}, {csv_filename}")
