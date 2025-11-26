# src/specific_search.py

from state import AgentState
from service import llm, retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List

# 1. State 포맷팅 (기존 로직 유지하되 간소화)
def format_state_to_string(state: AgentState) -> AgentState:
    formatted_lines = []
    keys_to_include = ['query', 'title', 'year', 'casts', 'director', 'genre', 'ott']

    for key in keys_to_include:
        value = state.get(key)
        if value:
            if isinstance(value, list):
                str_values = [str(v.value if hasattr(v, 'value') else v) for v in value]
                value_str = ", ".join(str_values)
            else:
                value_str = str(value)
            formatted_lines.append(f"{key}: {value_str}")

    context_str = "\n".join(formatted_lines) if formatted_lines else state['query']
    return {"rag_context": context_str}

# 2. 검색 쿼리 생성 (강력한 제약조건 추가)
rag_specialized_prompt_template = """
당신은 Vector DB 검색을 위한 **검색어 추출기**입니다.
주어진 정보를 바탕으로, DB에서 가장 정확한 문서를 찾을 수 있는 **단 하나의 검색 문장**을 만드세요.

[제약사항]
1. "검색어는 다음과 같습니다" 같은 말 절대 금지.
2. 오직 검색어 텍스트만 출력할 것.
3. 영화 제목이 있다면 제목을 가장 앞에 배치할 것.

[입력 정보]
{rag_context}

[출력 예시]
영화 승부 줄거리 출연진 넷플릭스 정보
"""

rag_query_prompt = ChatPromptTemplate.from_template(rag_specialized_prompt_template)
rag_query_generation_chain = (rag_query_prompt | llm | StrOutputParser())

def generate_rag_query(state: AgentState) -> AgentState:
    print("--- RAG: 검색 쿼리 생성 중 ---")
    rag_query = rag_query_generation_chain.invoke({"rag_context": state['rag_context']})
    print(f"최적화된 검색어: {rag_query}")
    return {"rag_query": rag_query}

# 3. 검색 (service.py의 retriever 사용)
def retrieve(state: AgentState) -> AgentState:
    rag_query = state.get('rag_query', state['query'])
    # Pinecone Retriever 호출
    docs = retriever.invoke(rag_query)
    return {'context': docs}

# 4. 답변 생성 (답변도 gpt-4o-mini로 빠르고 간결하게)
answer_prompt = """
당신은 영화/OTT 전문 큐레이터입니다. 
아래 [Context]에 있는 정보만을 사용하여 사용자의 질문에 답변하세요.

[Context]
{context}

[질문]
{question}

[지시사항]
1. Context에 없는 내용은 지어내지 말고 "죄송합니다, 해당 정보는 가지고 있지 않습니다"라고 하세요.
2. 답변은 친절하고 간결하게(3~4문장) 작성하세요.
3. OTT 정보가 있다면 반드시 언급해 주세요.
"""

rag_chain = (
    ChatPromptTemplate.from_template(answer_prompt)
    | llm
    | StrOutputParser()
)

def generate_answer(state: AgentState) -> AgentState:
    print("--- 답변 생성 중 ---")
    query = state['query']
    docs = state['context']
    
    # 문서 내용을 하나의 문자열로 합침
    context_str = "\n\n".join([doc.page_content for doc in docs])
    
    response = rag_chain.invoke({
        'question': query, 
        'context': context_str
    })
    
    return {'answer': response}