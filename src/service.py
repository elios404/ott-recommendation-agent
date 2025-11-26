# services.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# --- 1. LLM 및 Embedding 초기화 ---
llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings(model='text-embedding-3-large')


# --- 3. VectorStore (Pinecone) 연결 ---
# .env 파일에 PINECONE_API_KEY가 있어야 합니다.
api_key = os.getenv("PINECONE_API_KEY")
index_name = "movie-rag-index" # 생성한 인덱스 이름과 일치해야 함

if not api_key:
    raise ValueError("PINECONE_API_KEY가 설정되지 않았습니다.")

# Pinecone VectorStore 연결
vector_store = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# Retriever 설정 (검색 개수 k=5로 조금 늘려 정보량 확보)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})
