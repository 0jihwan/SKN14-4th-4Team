from PIL import Image

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout

from .inferer import OpenAIInferer, Inferer
from .utils import parse_prediction, get_menu_context_with_threshold, analyze_meal_with_llm, llm_extract_food_name
from .models import ChatHistory

def main(request):
    return render(request, 'main.html')

@csrf_exempt
def chat(request):
    is_auth = request.user.is_authenticated

    # 회원/비회원 모두 누적 menu_infos를 세션에서 관리
    if 'menu_infos' not in request.session:
        request.session['menu_infos'] = []
    menu_infos = request.session['menu_infos']

    error_message = None

    if request.method == 'POST':
        user_text = request.POST.get('user_text', '')
        files = request.FILES.getlist('uploaded_files')
        user_images = []

        for f in files[:5]:
            img = Image.open(f).convert('RGB')
            user_images.append(img)

        if user_images or user_text.strip():
            inferer = OpenAIInferer("gpt-4o-mini", 0.0)
            filenames = [f.name for f in files[:5]]
            results = inferer(user_images, filenames)

            # (1) 이미지 업로드시: LLM이 추출한 실제 음식명만 누적
            for filename, pred_str in results.items():
                menu_name, ingredients = parse_prediction(pred_str) 
                if menu_name:  # 음식명이 존재할 때만 누적!
                    rag_context, calorie = get_menu_context_with_threshold(menu_name)
                    menu_infos.append({
                        "filename": filename,
                        "menu_name": menu_name,
                        "calorie": calorie,
                        "ingredients": ingredients,
                        "rag_context": rag_context
                    })

            # (2) 텍스트 입력시: parse_prediction을 거쳐 음식명만 누적
            if not files and user_text:
                menu_name = llm_extract_food_name(user_text)
                ingredients = ""  # (혹시 필요하면 나중에 LLM 프롬프트로 재료도 추출 가능)
                if menu_name:
                    rag_context, calorie = get_menu_context_with_threshold(menu_name)
                    menu_infos.append({
                        "filename": "-",
                        "menu_name": menu_name,
                        "calorie": calorie,
                        "ingredients": ingredients,
                        "rag_context": rag_context
                    })
            # (3) 누적 menu_infos를 세션에 다시 저장
            request.session['menu_infos'] = menu_infos

            # (4) 누적 menu_infos 전체를 analyze_meal_with_llm에 전달
            result = analyze_meal_with_llm(menu_infos, user_text, None)

            # (5) 대화 기록 저장 (회원: DB, 비회원: 세션)
            if is_auth:
                ChatHistory.objects.create(user=request.user, role='user', content=user_text)
                ChatHistory.objects.create(user=request.user, role='assistant', content=result)
            else:
                if 'chat_history' not in request.session:
                    request.session['chat_history'] = []
                chat_history = request.session['chat_history']
                chat_history.append(('user', user_text))
                chat_history.append(('assistant', result))
                request.session['chat_history'] = chat_history

    # (6) 대화 내역 불러오기
    if is_auth:
        chat_history_objs = ChatHistory.objects.filter(user=request.user).order_by('created_at')
        chat_history = [(obj.role, obj.content) for obj in chat_history_objs]
    else:
        chat_history = request.session.get('chat_history', [])

    return render(request, 'chat.html', {'chat_history': chat_history})


def signup(request):
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        pw1 = request.POST.get('password1')
        pw2 = request.POST.get('password2')
        if pw1 != pw2:
            error = '비밀번호가 다릅니다.'
        elif User.objects.filter(username=username).exists():
            error = '이미 존재하는 아이디입니다.'
        else:
            user = User.objects.create_user(username=username, password=pw1)
            auth_login(request, user)
            return redirect('main')
    return render(request, 'signup.html', {'error': error})

def login_view(request):
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        pw = request.POST.get('password')
        user = authenticate(username=username, password=pw)
        if user:
            auth_login(request, user)
            return redirect('main')
        else:
            error = '아이디 또는 비밀번호가 틀렸습니다.'
    return render(request, 'login.html', {'error': error})

def logout_view(request):
    auth_logout(request)
    return redirect('main')


def clear_history(request):
    is_auth = request.user.is_authenticated

    # 누적 식사 기록(menu_infos) 초기화
    if 'menu_infos' in request.session:
        request.session['menu_infos'] = []
    if not is_auth:
        # 비회원: 세션의 대화 기록 초기화
        if 'chat_history' in request.session:
            request.session['chat_history'] = []
    else:
        # 회원: DB의 ChatHistory 삭제
        from .models import ChatHistory
        ChatHistory.objects.filter(user=request.user).delete()

    return redirect('chat')  # 혹은 메인화면 등 원하는 곳으로 리다이렉트
