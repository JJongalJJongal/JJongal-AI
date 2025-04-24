# 부기와 꼬기 - 아동 상호작용 챗봇 시스템

아이들의 창의성과 상상력을 자극하는 대화형 동화 생성 시스템입니다. 아이와 대화하며 관심사를 파악하고, 맞춤형 동화를 생성합니다.

## 시스템 구성

시스템은 두 개의 챗봇으로 구성되어 있습니다:

1. **부기(Chat-bot A)**: 아이와 대화를 나누며 관심사를 파악하고 이야기 주제를 추출합니다.
2. **꼬기(Chat-bot B)**: 추출된 주제를 바탕으로 상세 동화와 삽화, 음성을 생성합니다.

## 설치 방법

1. 프로젝트 클론
   ```bash
   git clone https://github.com/your-repo/CCB-AI.git
   cd CCB-AI
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

4. 환경 변수 설정
   - API 키 등 필요한 환경 변수를 `.env` 파일에 설정합니다.

## 사용 방법

### CLI 애플리케이션 실행

```bash
python chatbot/main.py
```

### 테스트 모드 실행

```bash
python chatbot/main.py --test
```

## 프로그램 흐름

1. 아이의 정보(이름, 나이, 관심사) 입력
2. 부기와 대화를 통해 아이의 관심사 및 선호도 파악
3. 대화 내용을 바탕으로 이야기 주제 추출
4. 꼬기가 추출된 주제로 상세 이야기 생성
5. 이야기에 맞는 삽화와 음성 생성 (선택 사항)
6. 생성된 결과물 저장 및 출력

## 디렉토리 구조

```
CCB-AI/
├── chatbot/
│   ├── models/
│   │   ├── chat_bot_a.py  # 부기(아이와 대화)
│   │   └── chat_bot_b.py  # 꼬기(이야기 생성)
│   ├── tests/
│   │   └── test_integration.py
│   └── main.py            # CLI 애플리케이션
├── output/                # 생성된 이야기, 삽화, 음성 파일 저장
├── data/                  # 학습 데이터, 프롬프트 등
├── requirements.txt
└── README.md
```

## 기술 스택

- Python 3.8+
- Natural Language Processing
- Large Language Models (LLM)
- Text-to-Image Generation
- Text-to-Speech

## 개발자

- [개발자 이름](mailto:email@example.com)

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요. 