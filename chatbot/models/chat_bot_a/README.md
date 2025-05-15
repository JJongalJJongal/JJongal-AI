# 부기 (ChatBot A) 사용 가이드

부기는 아이들과 대화하며 동화 줄거리를 수집하는 AI 챗봇입니다. 이 문서는 부기 챗봇의 설정 및 사용 방법을 안내합니다.

## 개요

부기 챗봇은 다음과 같은 특징을 가집니다:

- 아이와 친근한 대화를 통해 동화 줄거리 수집
- 아이의 관심사에 맞는 이야기 요소 제안
- 대화 흐름을 자연스럽게 유도하여 스토리 구성 요소 수집
- 연령에 맞는 대화 방식 사용
- 수집된 스토리는 챗봇 B(꼬기)에서 구체적인 동화로 개발

## 설치 및 초기화

### 초기화 코드 예시

```python
from CCB_AI.chatbot.models.chat_bot_a import StoryCollectionChatBot

# 챗봇 인스턴스 생성
chatbot = StoryCollectionChatBot()

# 아이 정보로 대화 초기화
greeting = chatbot.initialize_chat(
    child_name="민준",
    age=6,
    interests=["공룡", "우주", "로봇"],
    chatbot_name="부기"  # 기본값은 "부기"
)

print(greeting)  # 예: "안녕 민준아! 나는 부기야. 공룡을 좋아한다고 들었어!"
```

## 주요 기능

### 1. 대화 인터페이스

```python
# 사용자 입력에 대한 응답 생성
user_message = "나는 우주 비행사 이야기가 좋아!"
response = chatbot.get_response(user_message)
print(response)
```

### 2. 이야기 요소 제안

```python
# 이야기 요소 제안받기
suggestion = chatbot.suggest_story_element(user_message)
print(suggestion)
```

### 3. 스토리 주제 제안

```python
# 스토리 주제 제안 받기
theme_suggestion = chatbot.suggest_story_theme()
print(f"제안된 주제: {theme_suggestion['theme']}")
print(f"줄거리: {theme_suggestion['plot_summary']}")
```

### 4. 대화 요약 생성

```python
# 지금까지 나눈 대화 요약
conversation_summary = chatbot.get_conversation_summary()
print(conversation_summary)
```

### 5. 대화 저장 및 로드

```python
# 대화 저장
save_path = "conversations/민준_20240601.json"
chatbot.save_conversation(save_path)

# 대화 로드
chatbot.load_conversation(save_path)
```

## 토큰 사용량 관리

```python
# 토큰 사용량 확인
token_usage = chatbot.get_token_usage()
print(f"프롬프트 토큰: {token_usage['total_prompt']}")
print(f"완성 토큰: {token_usage['total_completion']}")
print(f"총 토큰: {token_usage['total']}")
```

## 모듈 구조

부기 챗봇은 다음과 같은 모듈로 구성되어 있습니다:

1. **story_collection_chatbot.py**: 챗봇의 메인 클래스 (StoryCollectionChatBot)
2. **message_formatter.py**: 메시지 형식 처리
3. **story_collector.py**: 이야기 요소 수집 로직
4. **conversation_manager.py**: 대화 관리 및 추적
5. **story_analyzer.py**: 스토리 내용 분석

## 아이 정보 업데이트

아이 정보는 언제든지 업데이트할 수 있습니다:

```python
# 아이 정보 업데이트
chatbot.update_child_info(
    child_name="민준", 
    age=6, 
    interests=["공룡", "우주", "로봇"]
)
```

## RAG 시스템 연동

부기 챗봇은 RAG(Retrieval-Augmented Generation) 시스템과 연동되어 있어 다양한 동화 지식에 기반한 대화가 가능합니다:

```python
# RAG 시스템을 통한 질의 예시
query_result = chatbot.rag_system.query("공룡에 관한 재미있는 동화가 있을까?")
print(query_result["answer"])
```

## 한국어 조사 처리

한국어 조사 처리를 위한 패턴을 사용합니다:

- "안녕 {name}아/야!" → "안녕 민준아!"
- "{name}이/가 좋아하는 것이 뭐야?" → "민준이 좋아하는 것이 뭐야?"

자세한 규칙은 MessageFormatter 클래스 내부를 참조하세요.

## 연령대별 접근 방식

- **4-6세**: 단순한 문장, 명확한 질문, 많은 격려
- **7-9세**: 더 복잡한 개념, 창의적 질문, 추론 유도

## 주의사항

1. 모든 메서드에 오류 처리가 포함되어 있지만, 필요에 따라 추가 예외 처리 필요
2. 토큰 제한 (기본 10,000)에 도달하면 대화가 제한될 수 있음
3. 오픈AI API 키 설정 필요 (shared.utils.openai_utils 참조) 