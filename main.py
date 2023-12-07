import os
import cv2
import csv
import math
import tempfile
import traceback
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import mediapipe as mp

static_path = os.path.join(os.path.dirname(__file__), "static")
templates_path = os.path.join(os.path.dirname(__file__), "templates")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=templates_path)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

inputWidth = 368
inputHeight = 368
inputScale = 1.0 / 255

scaler = StandardScaler()

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
    models = {
        1: {'name': 'lstm', 'path': 'joint_angle_classification_model_lstm.h5'},
        2: {'name': 'gru', 'path': 'joint_angle_classification_model_gru.h5'},
        3: {'name': 'simplernn', 'path': 'joint_angle_classification_model_simplernn.h5'}
    }

    predictions = {}

    for selected_model, model_info in models.items():
        model_name = model_info['name']
        model_path = model_info['path']

        model = keras.models.load_model(model_path, compile=False)

        capture = cv2.VideoCapture(video_path)

        csv_filename = f"joint_angle_video_model_{model_name}.csv"
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
            angles_test = df["Angle (Degrees)"]
            angles_test_normalized = scaler.fit_transform(angles_test.values.reshape(-1, 1))

            # 테스트 데이터에 대한 시퀀스 생성
            sequences_test = []
            sequence_length = 20
            for i in range(len(angles_test_normalized) - sequence_length + 1):
                sequence = angles_test_normalized[i:i + sequence_length]
                sequences_test.append(sequence)

            # 시퀀스를 Numpy 배열로 변환
            sequences_test = np.array(sequences_test)

            # 데이터를 입력(X)로 분할
            X_test = sequences_test[:, :-1]

            # 저장된 모델 불러오기
            loaded_model = keras.models.load_model(model_path, compile=False)

            # 테스트 데이터에 대한 예측 수행
            y_pred_proba = loaded_model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()

            # 0과 1의 비중 계산
            count_0 = np.sum(y_pred == 0)
            count_1 = np.sum(y_pred == 1)

            # 가장 많은 클래스 선택
            if count_0 > count_1:
                prediction = '정재서'
                percentage = count_0 / len(y_pred) * 100
                if percentage == 100:
                    image = 'js_100.jpg'
                elif percentage >= 90:
                    image = 'js_90.jpg'
                elif percentage >= 80:
                    image = 'js_80.jpg'
                elif percentage >= 70:
                    image = 'js_70.jpg'
                elif percentage >= 60:
                    image = 'js_60.jpg'
                elif percentage >= 50:
                    image = 'js_50.jpg'
            else:
                prediction = '주현상'
                percentage = count_1 / len(y_pred) * 100
                if percentage == 100:
                    image = 'hs_100.jpg'
                elif percentage >= 90:
                    image = 'hs_90.jpg'
                elif percentage >= 80:
                    image = 'hs_80.jpg'
                elif percentage >= 70:
                    image = 'hs_70.jpg'
                elif percentage >= 60:
                    image = 'hs_60.jpg'
                elif percentage >= 50:
                    image = 'hs_50.jpg'

            predictions[model_name] = {'prediction': prediction, 'percentage': percentage, 'image': image}

    return predictions


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


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_video/", response_class=HTMLResponse)
async def process_video(request: Request,
                        file: UploadFile = File(...)):
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_filename = temp_file.name
            temp_file.write(content)
            print(f"임시 파일 경로: {temp_filename}")

        predictions = process_video_content(temp_filename)
        os.remove(temp_filename)

        # 결과 HTML 파일 반환
        return templates.TemplateResponse(
            "video.html",
            {"request": request, "predictions": predictions}
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"예외 발생: {e}")
        # 추가: 예외 정보 출력
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error_message": str(e), "error_code": 15})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
