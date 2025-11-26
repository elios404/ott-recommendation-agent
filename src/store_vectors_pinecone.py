import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로드
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "movie-rag-index"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY가 .env 파일에 없습니다.")

# 2. 초기화
# 임베딩 모델: 차원 수 3072
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# Pinecone 클라이언트 초기화 (인덱스 확인용)
pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스가 없으면 생성 (보통 웹에서 만드는 게 더 확실하지만 코드에서도 가능)
if INDEX_NAME not in [index_info["name"] for index_info in pc.list_indexes()]:
    print(f"인덱스 '{INDEX_NAME}'가 없어 새로 생성합니다...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(2) # 생성 대기

# 3. 데이터 로드 및 Document 객체 변환
INPUT_FILE = "./output/rag_data_enhanced.jsonl" # 이전 단계에서 만든 파일
documents = []

print(f"'{INPUT_FILE}' 파일을 로드 중입니다...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Parsing JSONL"):
        item = json.loads(line)
        
        # (1) rag_text 가져오기 (이게 벡터화될 본문)
        page_content = item.get("rag_text", "")
        
        # (2) 메타데이터 매핑 (LangGraph 코드와 호환되도록 키 이름 변경)
        # 빈 리스트([])가 들어가면 에러가 날 수 있으므로, 없으면 None이나 빈 문자열 처리보다는
        # 빈 리스트를 그대로 두되, 검색 시 주의해야 합니다. Pinecone은 빈 리스트 허용합니다.
        
        metadata = {
            "title": item.get("title_ko", ""),
            "year": int(item.get("year")) if item.get("year") else 0,
            "genre": item.get("genres", []),          # genres -> genre
            "ott": item.get("ott_streaming_kr", []),  # ott_streaming_kr -> ott
            "casts": item.get("cast", []),            # cast -> casts
            "director": item.get("directors", []),    # directors -> director
            "rating": item.get("rating_kr", "정보없음"),
            "type": item.get("type", "movie")
        }
        
        # LangChain Document 객체 생성
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

print(f"총 {len(documents)}개의 문서가 준비되었습니다.")

# 4. Pinecone에 업로드 (배치 처리)
# 한 번에 너무 많이 보내면 타임아웃 될 수 있으므로 100개씩 끊어서 보냅니다.
BATCH_SIZE = 100

print("Pinecone에 벡터 저장을 시작합니다...")

vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding)

# tqdm으로 진행률 표시
for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="Uploading Batches"):
    batch = documents[i : i + BATCH_SIZE]
    try:
        vector_store.add_documents(batch)
    except Exception as e:
        print(f"배치 업로드 중 오류 발생 (index {i}): {e}")
        # 오류 발생 시 멈추지 않고 다음 배치 시도 (선택 사항)
        continue

print("\n✅ 모든 데이터가 Pinecone에 성공적으로 저장되었습니다!")
print(f"인덱스 이름: {INDEX_NAME}")