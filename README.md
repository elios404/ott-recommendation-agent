# RAG-AI_agent-Study

## 환경 활성화
1. uv venv
2. source .venv/bin/activate
3. uv pip install -r requirements.txt

.env 파일에 OPENAI_API_KEY 를 넣을 수 있도록

### 추가 라이브러리 설치 이후
uv pip freeze > requirements.txt

## chromaDB
chromaDB2 는 장르 원핫인코딩도 포함된 버전

## 실행 방법
```
python main_graph.py
```

```
streamlit run app.py
```

## 콘솔 접근
# [주의] IP 주소를 꼭 방금 확인한 새 주소로 바꿔서 입력하세요!
ssh -i "Movie-RAG-Bot.pem" ubuntu@새로운_IP_주소

docker run -d -p 8501:8501 --env-file .env --name my-bot movie-rag-bot

docker start my-bot

http://새로운_IP_주소:8501