# TensorFlow 2.17.0 이미지를 베이스로 사용
FROM tensorflow/tensorflow:2.17.0

# 작업 디렉토리 설정
WORKDIR /flask-ml-head

# 필요한 패키지를 설치하기 위한 requirements.txt 복사
COPY requirements.txt .

# pip으로 필요한 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 애플리케이션과 모델 파일을 복사
COPY . .

ENV FLASK_ENV="development"

# Flask 앱 실행
CMD ["python", "app.py"]
