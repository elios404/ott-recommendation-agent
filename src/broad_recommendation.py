# src/broad_recommendation.py

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import AgentState
# service.py에서 vector_store를 가져옵니다.
from service import llm, vector_store 

# --- 1. 광범위 추천 쿼리 생성 프롬프트 최적화 ---
broad_rag_prompt_template = """
당신은 영화/드라마 추천 전문가입니다.
제공된 [검색 조건]을 바탕으로, Vector DB에서 유사한 작품을 찾기 위한 **핵심 키워드 문장** 하나를 생성하세요.

[검색 조건]
{rag_context}

[지시사항]
1. "어떤 영화를 찾으시나요?" 같은 사족을 붙이지 마세요.
2. 오직 검색에 필요한 분위기, 장르, 키워드만 나열하세요.
3. 예시: "긴장감 넘치는 스릴러, 반전 있는 결말"
"""

broad_rag_prompt = ChatPromptTemplate.from_template(broad_rag_prompt_template)
broad_rag_chain = broad_rag_prompt | llm | StrOutputParser()

def generate_broad_rag_query(state: AgentState) -> AgentState:
    print("--- RAG: 광범위 추천 쿼리 생성 중 ---")
    rag_context = state.get('rag_context')
    
    if not rag_context or rag_context.strip() == "- (No specific query details provided) -":
        # 컨텍스트가 없으면 사용자 원본 쿼리 사용
        rag_query = state.get('query')
    else:
        rag_query = broad_rag_chain.invoke({"rag_context": rag_context})
    
    print(f"생성된 검색 쿼리: {rag_query}")
    return {"rag_query": rag_query}

# --- 2. Pinecone 메타데이터 필터링 ---

def _build_metadata_filter(state: AgentState) -> Optional[Dict[str, Any]]:
    """
    Pinecone용 메타데이터 필터를 생성합니다.
    데이터 저장 시 키 이름: genre, ott, casts, director, year, title, rating, type
    """
    filters = []
    
    # 1. 리스트형 필드 처리 (OR 조건)
    # 사용자가 'Action'과 'SF'를 원하면 -> genre가 Action이거나 SF인 것 검색
    # Pinecone에서 리스트 필드에 대한 $eq는 "contains"의 의미를 가집니다.
    list_fields_map = {
        'genre': 'genre',       # state key -> DB key
        'ott': 'ott',
        'casts': 'casts',
        'director': 'director'
    }

    for state_key, db_key in list_fields_map.items():
        values = state.get(state_key)
        if values:
            # Enum이나 객체인 경우 문자열로 변환
            str_values = [v.value if hasattr(v, 'value') else str(v) for v in values]
            
            if str_values:
                if len(str_values) == 1:
                    filters.append({db_key: {"$eq": str_values[0]}})
                else:
                    # 값이 여러 개면 $or 연산 (하나라도 포함되면 매칭)
                    or_conditions = [{db_key: {"$eq": v}} for v in str_values]
                    filters.append({"$or": or_conditions})

    # 2. 단일 값 필드 처리 (year)
    if state.get('year'):
        # year는 숫자형으로 저장했으므로 $eq (정확히 일치) 또는 $gte (이후) 등 사용 가능
        # 여기서는 편의상 $eq로 처리 (사용자 의도에 따라 수정 가능)
        filters.append({"year": {"$eq": state['year']}})
    
    if not filters:
        return None
    
    if len(filters) == 1:
        return filters[0]
    
    # 여러 필드 조건은 $and로 묶음 (예: 장르가 액션이면서 OTT가 넷플릭스인 것)
    return {"$and": filters}

def retrieve_with_filter(state: AgentState) -> AgentState:
    print("--- RAG: 메타데이터 필터링 검색 (Pinecone) ---")
    
    rag_query = state.get('rag_query')
    if not rag_query:
        rag_query = state['query']

    metadata_filter = _build_metadata_filter(state)
    
    search_kwargs = {'k': 5} # 검색 개수
    if metadata_filter:
        print(f"적용된 필터: {metadata_filter}")
        search_kwargs['filter'] = metadata_filter
    
    # Pinecone 검색
    docs = vector_store.similarity_search(
        query=rag_query,
        **search_kwargs
    )
    
    return {'context': docs}