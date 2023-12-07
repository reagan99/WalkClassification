import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import clear_output

# GPU를 사용할 경우 GPU 메모리 제한 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


def lr_scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# 기존 학습 데이터 로드
train_data_1 = pd.read_csv('js_1_mediapipe_angle.csv')
train_data_2 = pd.read_csv('js_2_mediapipe_angle.csv')
train_data_3 = pd.read_csv('js_3_mediapipe_angle.csv')
train_data_4 = pd.read_csv('js_4_mediapipe_angle.csv')
train_data_5 = pd.read_csv('js_5_mediapipe_angle.csv')
train_data_6 = pd.read_csv('js_6_mediapipe_angle.csv')
train_data_7 = pd.read_csv('js_7_mediapipe_angle.csv')
train_data_8 = pd.read_csv('js_8_mediapipe_angle.csv')
train_data_9 = pd.read_csv('js_9_mediapipe_angle.csv')
train_data_10 = pd.read_csv('hs_1_mediapipe_angle.csv')
train_data_11 = pd.read_csv('hs_2_mediapipe_angle.csv')
train_data_12 = pd.read_csv('hs_3_mediapipe_angle.csv')
train_data_13 = pd.read_csv('hs_4_mediapipe_angle.csv')
train_data_14 = pd.read_csv('hs_5_mediapipe_angle.csv')
train_data_15 = pd.read_csv('hs_6_mediapipe_angle.csv')
train_data_16 = pd.read_csv('hs_7_mediapipe_angle.csv')
train_data_17 = pd.read_csv('hs_8_mediapipe_angle.csv')
train_data_18 = pd.read_csv('hs_9_mediapipe_angle.csv')


# 학습 데이터 병합
train_data = pd.concat([train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6, train_data_7, train_data_8, train_data_9, train_data_10, train_data_11, train_data_12, train_data_13, train_data_14, train_data_15, train_data_16, train_data_17, train_data_18])

# 테스트 데이터 로드
test_data = pd.read_csv('hs_10_mediapipe_angle.csv')

# Feature 및 Label 선택
angles_train = train_data["Angle (Degrees)"]
angles_test = test_data["Angle (Degrees)"]

# (Z-score 정규화)
scaler = StandardScaler()
angles_train_normalized = scaler.fit_transform(angles_train.values.reshape(-1, 1))
angles_test_normalized = scaler.transform(angles_test.values.reshape(-1, 1))

# 시퀀스 생성
sequence_length = 20

# 학습 데이터 시퀀스 생성
sequences_train = []
for i in range(len(angles_train_normalized) - sequence_length + 1):
    sequence = angles_train_normalized[i:i + sequence_length]
    sequences_train.append(sequence)

# 테스트 데이터 시퀀스 생성
sequences_test = []
for i in range(len(angles_test_normalized) - sequence_length + 1):
    sequence = angles_test_normalized[i:i + sequence_length]
    sequences_test.append(sequence)

# 시퀀스 데이터를 Numpy 배열로 변환
sequences_train = np.array(sequences_train)
sequences_test = np.array(sequences_test)

# 데이터를 입력(X)와 출력(y)로 분할
X = sequences_train[:, :-1]  # 입력 시퀀스
y = sequences_train[:, -1]   # 출력(다음 각도 예측)

# Label 데이터 생성 (이진 분류를 위해 두 보행자를 0, 1로 인코딩)
labels_train = np.zeros(len(y))
labels_train[len(y)//2:] = 1

# 모델 정의
model = keras.Sequential()
model.add(layers.LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True))
model.add(layers.Dropout(0.4))
model.add(layers.LSTM(256, return_sequences=True))
model.add(layers.Dropout(0.4))
model.add(layers.LSTM(256))
model.add(layers.Dense(1, activation='sigmoid'))  # 1로 변경, 활성화 함수도 'sigmoid'로 변경
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# 얼리 스탑핑 콜백 설정
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# 모델 훈련
history = model.fit(X, labels_train, epochs=300, batch_size=16, validation_split=0.2, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler), early_stopping])

# 모델 저장
model.save('joint_angle_classification_model.h5')

# 테스트 데이터의 Label 생성 (이진 분류를 위해 두 보행자를 0, 1로 인코딩)
labels_test = np.zeros(len(sequences_test))
labels_test[len(sequences_test)//2:] = 1

# 테스트 데이터에 대한 예측
y_pred_proba = model.predict(sequences_test[:, :-1])
y_pred = np.argmax(y_pred_proba, axis=1)  # 확률 예측값을 클래스로 변환

# 결과 평가
accuracy = accuracy_score(labels_test, y_pred)
conf_matrix = confusion_matrix(labels_test, y_pred)
print(f"정확도: {accuracy}")
print("혼동 행렬:\n", conf_matrix)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(labels_test, label='실제 클래스')
plt.plot(y_pred, label='예측 클래스', linestyle='dashed')
plt.legend()
plt.title('실제 vs. 예측 클래스')
plt.show()

# 모델의 출력물과 입력으로 사용된 사용자의 일치율 출력
user_match_rate = accuracy_score(labels_test, y_pred_proba[:, 1] > 0.5)
print(f"모델의 출력물과 입력으로 사용된 사용자의 일치율: {user_match_rate}")
