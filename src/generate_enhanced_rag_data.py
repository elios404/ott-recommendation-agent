import json
import os
from tqdm import tqdm

# --- 설정 ---
# 입력 파일: TMDB API로 수집된 원본 데이터
INPUT_FILE_PATH = "./output/tmdb_data.jsonl"
# 출력 파일: RAG를 위해 텍스트가 보강된 데이터
OUTPUT_FILE_PATH = "./output/rag_data_enhanced.jsonl"

def build_rich_rag_text(item):
    """
    LLM이 검색하고 답변하기 최적화된 형태의 rag_text를 생성합니다.
    기존에 누락되었던 OTT 정보, 제작진, 관람 등급, 유형 등을 모두 포함합니다.
    """
    
    # 1. 기본 정보 (제목, 연도, 유형)
    title_ko = item.get('title_ko', '제목 없음')
    title_en = item.get('title_en', '')
    year = item.get('year', '')
    media_type = "영화" if item.get('type') == 'movie' else "TV 드라마/시리즈"
    
    # 2. 인물 정보 (감독, 작가, 배우)
    # 리스트가 비어있을 경우를 대비해 처리
    directors = ", ".join(item.get('directors', [])) if item.get('directors') else "정보 없음"
    writers = ", ".join(item.get('writers', [])) if item.get('writers') else "정보 없음"
    cast_list = item.get('cast', [])
    # 출연진은 너무 많으면 토큰 낭비가 심하므로 주요 10명까지만 끊습니다.
    cast = ", ".join(cast_list[:10]) if cast_list else "정보 없음"
    
    # 3. 장르 및 등급
    genres = ", ".join(item.get('genres', [])) if item.get('genres') else "정보 없음"
    rating = item.get('rating_kr', '정보 없음')
    if not rating: rating = "정보 없음" # 빈 문자열일 경우 처리

    # 4. OTT 정보 (핵심 요청 사항)
    # 스트리밍, 대여, 구매 정보를 명확히 구분하여 적어줍니다.
    streaming = item.get('ott_streaming_kr', [])
    rent = item.get('ott_rent_kr', [])
    buy = item.get('ott_buy_kr', [])
    
    ott_text_parts = []
    if streaming:
        ott_text_parts.append(f"스트리밍: {', '.join(streaming)}")
    if rent:
        ott_text_parts.append(f"대여: {', '.join(rent)}")
    if buy:
        ott_text_parts.append(f"구매: {', '.join(buy)}")
        
    if not ott_text_parts:
        ott_info = "현재 한국에서 이용 가능한 OTT 정보가 없습니다."
    else:
        ott_info = " / ".join(ott_text_parts)

    # 5. 줄거리 및 키워드
    overview = item.get('overview', '')
    keywords = ", ".join(item.get('keywords', [])) if item.get('keywords') else ""

    # --- rag_text 조합 ---
    # LLM이 읽기 좋게 'Key: Value' 형태로 명확히 구조화합니다.
    text_parts = [
        f"[제목] {title_ko}",
        f"[영문 제목] {title_en}",
        f"[제작 연도] {year}",
        f"[유형] {media_type}",
        f"[장르] {genres}",
        f"[관람 등급] {rating}",
        f"[감독] {directors}",
        f"[작가/각본] {writers}",
        f"[주요 출연진] {cast}",
        f"[OTT 서비스(한국)] {ott_info}",
        f"[줄거리] {overview}",
        f"[키워드] {keywords}"
    ]
    
    # 빈 줄을 제거하고 하나의 문자열로 합칩니다.
    return "\n".join(part for part in text_parts if part)

def main():
    # 파일 존재 여부 확인
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"오류: 입력 파일 '{INPUT_FILE_PATH}'을 찾을 수 없습니다.")
        return

    print(f"Processing '{INPUT_FILE_PATH}'...")
    
    processed_count = 0
    
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
        
        # 전체 라인 수 계산 (tqdm 진행률 표시용)
        infile.seek(0)
        total_lines = sum(1 for _ in infile)
        infile.seek(0)
        
        for line in tqdm(infile, total=total_lines, desc="Generating RAG Data"):
            try:
                if not line.strip(): continue
                
                item = json.loads(line)
                
                # 강화된 rag_text 생성
                rich_rag_text = build_rich_rag_text(item)
                
                # 원본 데이터에 rag_text 덮어쓰기 (또는 추가)
                item['rag_text'] = rich_rag_text
                
                # 파일에 쓰기
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                processed_count += 1
                
            except json.JSONDecodeError:
                print("JSON 디코딩 에러 발생. 해당 라인을 건너뜁니다.")
                continue
            except Exception as e:
                print(f"처리 중 오류 발생: {e}")
                continue

    print(f"\n완료되었습니다!")
    print(f"총 {processed_count}개의 항목이 처리되어 '{OUTPUT_FILE_PATH}'에 저장되었습니다.")
    print("이제 이 파일을 사용하여 Vector DB(Pinecone 등)에 임베딩을 진행하시면 됩니다.")

if __name__ == "__main__":
    main()