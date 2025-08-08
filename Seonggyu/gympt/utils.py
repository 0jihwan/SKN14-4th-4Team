from typing import List, Tuple, Dict, Any
import openai

from langchain_openai import ChatOpenAI

import os
import re
import ast

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


PINECONE_PJ_KEY     = os.environ.get("PINECONE_PJ_KEY")
INDEX_NAME          = "food-index"
EMBED_MODEL         = "text-embedding-3-small"

pc                  = Pinecone(api_key=PINECONE_PJ_KEY)
index               = pc.Index(INDEX_NAME)

embeddings          = OpenAIEmbeddings(model=EMBED_MODEL)
vector_store        = PineconeVectorStore(index=index, embedding=embeddings)


# OpenAI가 반환한 문자열 응답을 (음식명, 재료) 형태로 파싱하는 함수
# 문자열이 [("짜장면", "춘장, 면, 돼지고기")] 형태일 경우 → 튜플로 변환
def parse_prediction(pred_str: str) -> Tuple[str, str]:
    try:
        parsed = ast.literal_eval(pred_str)
        menu_name, ingredients = parsed[0]
        return menu_name.strip(), ingredients.strip()
    except:
        return "", ""
    

def llm_extract_food_name(text: str) -> str:
    """
    LLM에게 '이 문장에서 음식명만 한글로 알려줘!'라고 요청
    """
    prompt = f"아래 문장에서 음식명만 한글로 한 개만, 다른 말 없이 순수 텍스트로 답하세요.\n문장: '{text}'\n음식명:"
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# Pinecone 유사도 검색 결과를 문자열로 변환해서 LLM에게 줄 context로 정리 (예: 메뉴명, 칼로리, 유사도)
def build_context(matches: List[Tuple]) -> str:
    lines = []
    for doc, score in matches:
        meta = doc.metadata or {}
        name = meta.get("RCP_NM", "알 수 없는 메뉴")
        kcal = meta.get("INFO_ENG", "칼로리 정보 없음")
        lines.append(f"- 메뉴명: {name}, 칼로리: {kcal} (유사도: {score:.2f})")
    return "\n".join(lines)


# 음식명만 있을 경우, OpenAI에게 직접 칼로리 숫자를 요청
# Pinecone 유사도가 낮을 때 fallback 용도로 사용
def ask_llm_calorie(menu_name: str) -> str:
    try:
        prompt = f"다음 음식의 대표적인 1인분 칼로리(kcal) 숫자만 알려주세요 **반드시 숫자만 반환!!**: '{menu_name}'"
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except:
        return "250"  # fallback
    

def search_menu(menu_name: str, k: int = 3):
    return vector_store.similarity_search_with_score(query=menu_name, k=k)


def get_menu_context_with_threshold(
    menu_name: str,
    k: int = 1,
    threshold: float = 0.4
) -> Tuple[str, str]:
    
    if not menu_name or menu_name.strip() == "":
        return "", ""

    matches = search_menu(menu_name, k)

    if not matches or matches[0][1] < threshold:
        calorie = ask_llm_calorie(menu_name)
        context = f"- 메뉴명: {menu_name}, 칼로리: {calorie}"
        return context, calorie

    context = build_context(matches)
    doc, _ = matches[0]
    calorie = doc.metadata.get("INFO_ENG")

    if not calorie or not str(calorie).isdigit():
        calorie = ask_llm_calorie(menu_name)

    return context, calorie


def strip_codeblock(text):
    # ```html ... ``` 또는 ``` ... ``` 감싸진 부분 제거
    return re.sub(r"^```(?:html)?\s*([\s\S]*?)\s*```$", r"\1", text.strip())


def analyze_meal_with_llm(menu_infos, user_info, rag_context="", chat_history=None) -> str:
    prompt_tmpl = """
아래 [오늘 섭취한 음식 정보]와 [사용자 정보]를 참고해 HTML로만 답변하세요.
"어떤 경우에도 <table>, <ul> 등 HTML 구조를 지킬것"

[오늘 섭취한 음식 정보]
{foods_context}
{table}
<b>총 섭취 칼로리: {total_calorie}kcal</b>

아래는 지금까지의 대화 내역입니다.
{history_prompt}

사용자 정보: {user_info}

[답변 지침]

1. 입력 음식 정보(menu_infos)가 '없거나 비어있으면':
    - <b>표, 운동/식단 추천을 하지 말고</b>, 음식 정보를 입력해 달라는 안내 메시지(자연어)만 반환할 것.
    - 예시: "음식 정보를 입력해주시면 섭취 칼로리 및 맞춤형 운동/식단 추천을 도와드릴 수 있습니다."
    - 만약 음식이 아닌 다른 정보가 올 경우, 당신은 음식 & 운동 전용 챗봇임을 알리고, 다시 입력해 줄 것을 요청할 것. 

2. 음식 정보가 1개 이상 존재하면:
    - <table>, <ul> 등 HTML로 보기 좋게 작성
    - 표 구조, 음식/칼로리/운동/식단 추천 모두 위의 안내대로 작성
    - 답변을 'HTML코드를 이용해' 예쁘게 작성할 것. (예: 표는 <table>, 운동추천은 <ul><li>운동명</li></ul>)
    - 표 구조를 지킬것: <table><tr><th>No</th>...</tr><tr>...</tr></table>
    - 먹은 음식(여러 개면 모두)(이미지로 받은 음식과, 텍스트로 받은 정보의 음식 모두)과 각각의 칼로리 정보를 HTML 표로 보여줄 것 (위 예시 참고)
    - 모든 음식의 총 섭취 칼로리를 계산해서 보여줄 것. 칼로리 총합은 <b>태그로 굵게</b> 강조
    - 사용자의 신체 정보와 운동량을 고려하여 1일 권장 섭취량을 계산하고 보여줄 것
    - 사용자의 신체 정보와 운동량이 입력 되지 않았다면, 평균 성인 남성의 대사량을 기반으로 계산할 것.
    - 1일 권장 섭취량과 남은 칼로리 계산

    - 사용자가 섭취한 칼로리를 소모할 수 있는 운동 추천(<ul>) (추천 운동의 '칼로리' 합계 = 총 섭취 칼로리) 목록으로 보기 좋게 HTML로 작성
    - 운동은 운동 종목과 소요시간, 소비 칼로리를 함께 알려줄 것.
    - 운동은 기본적으로 유산소 운동 1개, 근력운동 1개, 구기종목 1개를 제시해서 주천해줄 것.
    - 사용자가 추가적으로 운동 방향을 제시하면 그 방향에 맞춰 추천해줄 것.
    - 모든 답변은 한 번에, 보기 좋게 작성할 것
    - 예시와 같이 HTML 태그를 쓸것.

    [식단 추천 규칙 - 반드시 지킬 것]
    - 아래 표의 '칼로리' 합계는 n = (권장섭취량 - {total_calorie})kcal와 정확히 일치해야 한다. 1kcal라도 다르면 무효!
    - 끼니는 반드시 "아침/점심/간식/저녁" 등 남은 끼니를 표시할 것.
    - 메뉴는 실제 음식만 제시. 양(단위, 인분/개/알/gram) 필수. kcal는 정확한 숫자(소수점X)
    - <table>을 써서 행마다 '끼니','메뉴','양','칼로리' 네 칸.
    - 한 줄 요약X. 표만! 표 아래 칼로리 총합, 남은 칼로리 표기, *위 식단의 칼로리 총합은 n kcal로 남은 칼로리와 다르면 절대 안됨* .
    - 식단의 방향을 지정해주면 그 방향에 맞춰 다시 식단 추천 (예시: 벌크업 / 체중감소 / 비건/ 저탄수화물 고지방 식단 등)

    [좋은 예시]
    <table border="1">
    <tr><th>끼니</th><th>메뉴</th><th>양</th><th>칼로리</th></tr>
    <tr><td>점심</td><td>닭가슴살</td><td>150g</td><td>200</td></tr>
    <tr><td>간식</td><td>아몬드</td><td>30g</td><td>170</td></tr>
    <tr><td>저녁</td><td>야채 샐러드</td><td>200g</td><td>100</td></tr>
    </table>
    식단 총합: 470kcal (n kcal)

"""
    try:
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
        history_prompt = ""
        if chat_history:
            for i, (role, content, images) in enumerate(chat_history[-5:]):
                who = "사용자" if role == "user" else "GYM-PT"
                history_prompt += f"{who}: {content}\n"

        # 여러 음식 정보 표와 요약 만들기
        table = '<table border="1"><tr><th>No</th><th>파일명</th><th>음식명</th><th>칼로리</th></tr>'
        for i, info in enumerate(menu_infos):
            menu = info.get("menu_name", "")
            kcal = info.get("calorie", "")
            filename = info.get("filename", "")
            table += f'<tr><td>{i+1}</td><td>{filename}</td><td>{menu}</td><td>{kcal}</td></tr>'
        table += '</table>'
        foods_context = ""
        total_calorie = 0
        for i, info in enumerate(menu_infos):
            menu = info.get("menu_name", "")
            kcal = info.get("calorie", "")
            filename = info.get("filename", "")
            foods_context += f"{i + 1}. {filename}: {menu} ({kcal}kcal)\n"
            try:
                total_calorie += int(float(kcal))
            except:
                pass

        prompt = prompt_tmpl.format(foods_context=foods_context, table=table, total_calorie=total_calorie, history_prompt=history_prompt, user_info=user_info)
        result = llm.invoke(prompt)
        cleaned_result = strip_codeblock(result.content)
        return cleaned_result

        print("증강된 프롬프트", "-"*40, cleaned_result, sep="\n", end="\n")
        # print("최종 결과 답변", "-"*40, result.content, sep="\n", end="\n")

    except Exception as e:
        return f"분석 중 오류가 발생했습니다: {str(e)}"