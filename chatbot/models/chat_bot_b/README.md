# 꼬기 (ChatBot B) 사용 가이드

꼬기는 동화 줄거리를 바탕으로 일러스트와 내레이션을 생성하는 AI 챗봇입니다. 이 문서는 꼬기 챗봇의 설정 및 사용 방법을 안내합니다.

## 개요

꼬기 챗봇은 다음과 같은 특징을 가집니다:

- 부기(ChatBot A)에서 수집된 줄거리로 완성된 동화 생성
- DALL-E를 활용한 아름다운 일러스트 생성
- ElevenLabs로 생성된 고품질 내레이션 오디오
- 연령에 맞는 교육적 가치와 내용 조정
- 일관된 스토리와 캐릭터 구성 유지

## 설치 및 초기화

### 초기화 코드 예시

```python
from CCB_AI.chatbot.models.chat_bot_b import StoryGenerationChatBot

# 챗봇 인스턴스 생성 (기본 출력 경로는 "output")
chatbot = StoryGenerationChatBot(output_dir="output")

# 동화 줄거리 정보 설정
story_outline = {
    "theme": "우주 모험",
    "characters": ["용감한 우주 비행사 지훈", "로봇 친구 알피"],
    "setting": "미래의 우주 정거장",
    "plot_summary": "지훈이는 작은 우주선을 타고 미지의 행성으로 모험을 떠납니다. 도중에 문제가 생기지만 로봇 친구 알피의 도움으로 무사히 해결합니다.",
    "educational_value": "용기, 우정, 문제 해결 능력",
    "target_age": 6
}

# 스토리 아웃라인 설정
chatbot.set_story_outline(story_outline)

# 대상 연령 설정 (필요시)
chatbot.set_target_age(6)
```

## 주요 기능

### 1. 상세 스토리 생성

```python
# 상세 스토리 생성
detailed_story = chatbot.generate_detailed_story()

# 결과 출력
print(f"챕터 수: {len(detailed_story['chapters'])}")
for chapter in detailed_story['chapters']:
    print(f"제목: {chapter['title']}")
    print(f"내용: {chapter['narration'][:100]}...")
```

### 2. 일러스트 생성

```python
# 일러스트 생성
image_paths = chatbot.generate_illustrations()

# 결과 출력
print(f"생성된 이미지 수: {len(image_paths)}")
for path in image_paths:
    print(f"이미지 경로: {path}")

# 특정 장면에 대한 이미지 생성
scene_description = "지훈이가 반짝이는 별들 사이로 우주선을 타고 날아가는 모습"
image_path = chatbot.generate_image(scene_description)
print(f"이미지 경로: {image_path}")
```

### 3. 음성 내레이션 생성

```python
# 음성 내레이션 생성 (비동기 함수)
import asyncio

# asyncio 이벤트 루프 실행
audio_files = asyncio.run(chatbot.generate_voice())

# 결과 출력
print(f"생성된 오디오 파일: {audio_files}")
```

### 4. 스토리 데이터 저장 및 로드

```python
# 스토리 데이터 저장
save_path = "stories/우주모험_20240601.json"
chatbot.save_story_data(save_path)

# 스토리 데이터 로드
chatbot.load_story_data(save_path)
```

### 5. 동화 미리보기 생성

```python
# 동화 미리보기 생성
preview = chatbot.get_story_preview()
print(f"제목: {preview['title']}")
print(f"미리보기: {preview['preview']}")
```

## 전체 동화 생성 워크플로

```python
# 완전한 동화 생성 워크플로
images, narration = chatbot.generate_story()

print(f"생성된 이미지: {len(images)}개")
print(f"내레이션 오디오 파일: {narration}")
```

## 모듈 구조

꼬기 챗봇은 다음과 같은 모듈로 구성되어 있습니다:

1. **story_generation_chatbot.py**: 챗봇의 메인 클래스 (StoryGenerationChatBot)
2. **content_generator.py**: 텍스트, 이미지, 오디오 생성 로직
3. **story_parser.py**: 스토리 구조 분석 및 파싱
4. **media_manager.py**: 이미지와 오디오 파일 관리
5. **data_persistence.py**: 데이터 저장 및 로드 기능

## RAG 시스템 연동

꼬기 챗봇은 RAG(Retrieval-Augmented Generation) 시스템과 연동되어 있어 더 풍부한 동화 내용 생성이 가능합니다:

```python
# RAG 시스템을 통한 콘텐츠 강화 예시
similar_stories = chatbot.rag_system.get_similar_stories(
    theme="우주 모험",
    age_group=6,
    n_results=3
)

for story in similar_stories:
    print(f"참고 스토리: {story['title']}")
    print(f"유사도: {story['similarity']}")
```

## 멀티미디어 자산 관리

꼬기 챗봇은 다음과 같은 형식으로 멀티미디어 자산을 관리합니다:

### 이미지 파일

- 형식: JPEG
- 해상도: 1024x1024
- 저장 경로: `output/images/{story_id}_{chapter_number}.jpg`

### 오디오 파일

- 형식: MP3
- 샘플링 레이트: 24kHz
- 저장 경로: `output/audio/{story_id}_{chapter_number}.mp3`

## 주의사항

1. OpenAI API와 ElevenLabs API 키가 필요합니다
2. 이미지 및 오디오 생성에는 API 사용량 비용이 발생할 수 있습니다
3. 한 번에 너무 많은 이미지나 오디오를 생성하면 API 제한에 걸릴 수 있습니다
4. 이미지 생성 프롬프트 최적화를 위해 스타일 가이드 참조 필요
5. 출력 디렉토리는 사용 전 초기화하는 것이 좋습니다 