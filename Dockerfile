# 1. 베이스 이미지 설정 (Python 3.11 가벼운 버전 사용)
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치
# 캐시 효율을 위해 소스코드보다 먼저 복사합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
# src 폴더를 컨테이너 내부의 /app/src 로 복사합니다.
COPY src/ ./src/

# 5. Streamlit 포트 노출
EXPOSE 8501

# 6. 헬스체크 (선택 사항: 앱이 잘 떠있는지 확인)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 7. 실행 명령어
# .env 파일은 이미지에 포함하지 않고, 컨테이너 실행 시 주입받습니다.
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]