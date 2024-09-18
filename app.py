from flask import Flask, request, jsonify, logging
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.utils import img_to_array
import imageio
import io
import requests

app = Flask(__name__)

# H5 파일로부터 모델 로드
MODEL_PATH = 'model/head-alexnet-split-layer-2.h5'  # Head 모델의 H5 파일
head_model = tf.keras.models.load_model(MODEL_PATH)

# Tail URL
TAIL_URL = "http://tail-service/tail_predict"

# 이미지 전처리 함수
def preprocess_image(image):
    image = image.resize((250, 250)).convert('RGB')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Head 모델에서 Tail로 중간 결과 전송 및 최종 결과 반환 API 엔드포인트
@app.route('/head_predict_and_forward', methods=['POST'])
def head_predict_and_forward():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    app.logger.debug("File received:", file.filename)

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        app.logger.debug("Image processed")
        
        # Head 모델 예측 수행
        head_output = head_model.predict(processed_image)
        app.logger.debug(f"Head model output: {head_output}")  # 로그 추가
        
        # 중간 결과를 Tail 모델로 전송하고 최종 결과 반환
        tail_response = send_output_to_tail(head_output)

        if tail_response is None:  # Tail 모델이 응답하지 않으면
            return jsonify({'error': 'Failed to reach Tail model'}), 500

        if tail_response.status_code == 200:
            return jsonify(tail_response.json())
        else:
            return jsonify({'error': 'Tail model prediction failed'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def send_output_to_tail(head_output):
    json_data = {'head_output': head_output.tolist()}
    app.logger.debug(f"Sending data to Tail: {head_output.shape}")
    try:
        response = requests.post(TAIL_URL, json=json_data, timeout=10)
        app.logger.debug(f"Response from Tail: {response.status_code}, {response.text}")
        return response
    except requests.exceptions.RequestException as e:
        app.logger.debug(f"Request failed: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.logger.setLevel(10)
    app.run(host='0.0.0.0', port=8080, debug=True)
