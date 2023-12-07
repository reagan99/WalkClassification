import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def predict(file_path):
    # 테스트 데이터 로드
    test_data = pd.read_csv(file_path)

    # Feature 선택
    angles_test = test_data["Angle (Degrees)"]

    # Z-score 정규화
    scaler = StandardScaler()
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
    loaded_model = keras.models.load_model('joint_angle_classification_model_lstm.h5', compile=False)

    # 테스트 데이터에 대한 예측 수행
    y_pred_proba = loaded_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # 0과 1의 비중 계산
    count_0 = np.sum(y_pred == 0)
    count_1 = np.sum(y_pred == 1)

    # 가장 많은 클래스 선택
    if count_0 > count_1:
        prediction = 0
        percentage = count_0 / len(y_pred) * 100
    else:
        prediction = 1
        percentage = count_1 / len(y_pred) * 100

    # 파일 이름, 예측된 클래스, 예측된 클래스의 비중 리턴
    return file_path, prediction, percentage

# 여러 데이터 파일에 대해 실행
file_paths = [

    'joint_angle_video.csv'
]

# 결과를 저장할 리스트
results = []

for file_path in file_paths:
    result = predict(file_path)
    results.append(result)

# 결과 출력
for file_name, prediction, percentage in results:
    print(f"{file_name} : {prediction} (예측 비중: {percentage:.2f}%)")
