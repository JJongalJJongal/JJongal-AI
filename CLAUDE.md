# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CCB_AI is an EduTech project called "쫑알쫑알" (JjongAlJjongAl/Chattering) that creates interactive fairy tale books from children's imagination using AI. The system consists of two main AI chatbots working together:

- **ChatBot A (쫑이/Jjongi)**: Voice-interactive story collection bot that gathers story elements from children through real-time conversation
- **ChatBot B (아리/Ari)**: Story completion bot that generates complete multimedia fairy tales with text, images, and voice

## Development Commands

### Docker-based Development
```bash
# Build and run development environment
make dev                    # Start development environment with live reload
make build                  # Build Docker images
make run                    # Start production environment
make stop                   # Stop all services

# Demo and testing
make demo-setup            # Complete demo preparation (env check + build + run)
make demo-start            # Quick demo start
make demo-status           # Check service status
make demo-test             # Run functionality tests
make health                # Health check

# Monitoring and debugging
make logs                  # View all logs
make logs-ai              # View AI server logs only
make shell                # Access container shell
make monitor              # Real-time performance monitoring
```

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest chatbot/tests/ -v

# Direct server start (development)
cd chatbot
python app.py
```

### API Testing
```bash
# API 문서 확인 (Swagger UI)
http://localhost:8000/docs

# API 문서 확인 (ReDoc)
http://localhost:8000/redoc

# 개발용 JWT 토큰 생성
curl -X POST "http://localhost:8000/api/v1/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "dev_user_001", "password": "dev"}'

# WebSocket 연결 테스트
# wss://localhost:8000/wss/v1/audio?token=YOUR_JWT_TOKEN
```

### Vector Database Management
```bash
# Initialize RAG system with sample data
python -c "from chatbot.data.vector_db.core import VectorDB; VectorDB().initialize_with_samples()"

# Populate vector database
python chatbot/data/vector_db/populate_vector_db.py

# Manage vector database
python chatbot/data/vector_db/manage_vector_db.py
```

## Architecture Overview

### Core Components
- **FastAPI Application** (`chatbot/app.py`): Main server with WebSocket endpoints
- **ChatBot A (쫑이)** (`chatbot/models/chat_bot_a/`): Story collection via voice interaction
- **ChatBot B (아리)** (`chatbot/models/chat_bot_b/`): Complete story generation with multimedia
- **Vector Database** (`chatbot/data/vector_db/`): LangChain + ChromaDB for RAG system
- **Workflow System** (`chatbot/workflow/`): Orchestrates the complete story generation pipeline

### API Architecture Overview

#### **Integrated API Structure**
```
External Backend API ←→ JJongal AI Service ←→ Client App
   (User/Community)        (AI/Story Generation)
```

#### **JJongal AI REST API** (`/api/v1/*`)
- `POST /api/v1/auth/external-login` - 외부 토큰으로 내부 인증
- `POST /api/v1/auth/refresh` - 내부 토큰 갱신
- `GET /api/v1/users/me` - 외부 API에서 사용자 정보 조회
- `POST /api/v1/stories/generate` - AI 동화 생성 + 커뮤니티 공유
- `GET /api/v1/stories` - 생성된 동화 목록
- `GET /api/v1/stories/{story_id}` - 동화 상세 조회
- `PATCH /api/v1/stories/{story_id}` - 동화 정보 수정
- `DELETE /api/v1/stories/{story_id}` - 동화 삭제
- `GET /api/v1/community/stories` - 커뮤니티 동화 조회

#### **External API Integration** (백엔드 연동)
- `POST /user/signup` - 회원가입 (외부 처리)
- `POST /user/login` - 로그인 (외부 처리)
- `POST /oauth/{provider}` - 소셜 로그인 (외부 처리)
- `POST /board/create` - 동화 커뮤니티 공유 (자동 연동)
- `GET /board/read` - 커뮤니티 동화 목록 (연동 조회)

#### **WebSocket API** (`/wss/v1/audio`)
- Connection: `wss://domain/wss/v1/audio?token=JWT_TOKEN`
- Message Types: `start_conversation`, `user_message`, `end_conversation`
- Binary Audio: Voice cloning sample collection (5 samples → instant clone)
- Server Events: `conversation_started`, `ai_response`, `voice_clone_progress`
- Real-time Features: STT → ChatBot A → Voice Clone → TTS

#### **Authentication Flow**
```
1. User login via External API → External Token
2. External Token → JJongal AI → Internal JWT Token  
3. API Calls: Authorization: Bearer {internal_token} + X-External-Token: {external_token}
4. WebSocket: wss://domain/wss/v1/audio?token={internal_token}
```

### Key Features
1. **Real-time Voice Cloning**: Automatically collects and clones child's voice during conversation
2. **RAG System**: LangChain-based retrieval system using existing fairy tale knowledge
3. **Multimedia Generation**: DALL-E 3 images, ElevenLabs voice synthesis
4. **Age-Appropriate Content**: Specialized prompts for 4-7 and 8-9 age groups
5. **WebSocket Streaming**: Real-time audio processing with binary data transfer

### Technology Stack
- **Backend**: FastAPI, WebSockets, External API Integration
- **AI/ML**: OpenAI (GPT-4, DALL-E 3, Whisper), ElevenLabs, LangChain
- **Vector DB**: ChromaDB with OpenAI embeddings (RAG knowledge base)
- **Audio**: RNNoise for noise reduction, real-time voice cloning
- **External Integration**: HTTP Client (aiohttp), JWT validation
- **Deployment**: Docker, Docker Compose, Nginx

### Data Flow
```
[External API: User/Auth] ←→ [JJongal AI Service] ←→ [ChromaDB: RAG Knowledge]
        ↓                           ↓                        ↓
[Story Metadata Storage]    [AI Processing Pipeline]   [Vector Embeddings]
        ↓                           ↓                        ↓
[Community Board]      [WebSocket: Real-time Chat]    [Story Enhancement]
                               ↓
                    [쫑이→아리: Complete Story] → [Multimedia Output]
                               ↓
                    [Save to External API] → [Optional Community Sharing]
```

### Configuration
- **Environment Variables**: Defined in `.env` file
  - `EXTERNAL_API_URL`: Backend API endpoint
  - `EXTERNAL_API_KEY`: Backend API authentication
  - `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`: AI service keys
  - `CHROMA_DB_PATH`: ChromaDB vector database path
- **App Config**: `shared/configs/app_config.py` - Central configuration management
- **Prompts**: `shared/configs/prompts_config.py` - Age-specific conversation prompts

### Testing
- **Test Suite**: `chatbot/tests/test_chatbot.py`
- **Manual Testing**: WebSocket test endpoints and demo commands available
- **Health Checks**: Built-in health monitoring at `/api/v1/health`

### Data Storage Architecture
```
External API (SQL DB)          JJongal AI Service           ChromaDB (Vector)
├── users/                    ├── temp/                    ├── embeddings/
├── stories/                  │   ├── audio_samples/       ├── fairy_tale_kb/
├── children_profiles/        │   ├── generated_images/    └── rag_documents/
├── community_posts/          │   └── temp_audio/
└── authentication/           └── sessions/ (memory)
```

### Persistent vs Temporary Data
- **External API**: All permanent data (users, stories, community)
- **JJongal AI**: Temporary processing data, active sessions
- **ChromaDB**: Vector embeddings for RAG knowledge base

### Important Notes
- **Voice Cloning**: Requires 5 audio samples before cloning activation
- **Binary Audio Transfer**: WebSocket audio endpoints expect binary data, not JSON
- **Age-Based Processing**: Different prompt strategies for 4-7 vs 8-9 age groups
- **Memory Management**: Persistent conversation storage with SQLite + LangChain
- **Resource Limits**: Docker containers configured with memory limits (8GB for AI services)

### Common Development Tasks
- **Adding REST API Endpoints**: Modify `chatbot/api/v1/routers/api_routers.py`
- **Adding New Story Elements**: Modify `chatbot/models/chat_bot_a/core/story_engine.py`
- **Updating Voice Processing**: Edit `chatbot/models/voice_ws/` components  
- **RAG System Changes**: Work with `chatbot/data/vector_db/` modules
- **WebSocket Modifications**: Update `chatbot/api/v1/ws/` handlers
- **Chatbot Personality Updates**: Edit prompts in `shared/configs/prompts_config.py`
- **Vector Database Management**: Use `chatbot/data/vector_db/manage_vector_db.py`
- **Authentication System**: Configure JWT settings in `chatbot/api/v1/dependencies.py`
- **Data Models**: Update Pydantic models in `chatbot/api/v1/models.py`

### API Development Guidelines
- **REST API**: Follow FastAPI conventions with proper error handling and response models
- **Authentication**: JWT-based authentication for both REST and WebSocket APIs
- **Response Format**: Standardized JSON responses with success/error structure
- **WebSocket**: Mixed JSON/Binary protocol for real-time audio processing
- **Rate Limiting**: Implement appropriate limits for API endpoints
- **Documentation**: Auto-generated via FastAPI/Swagger at `/docs` endpoint

### Project Evolution Notes
- **Service rebranded** from "꼬꼬북" to "쫑알쫑알" (legacy references may still exist in some files)
- **Recent major refactoring** focused on WebSocket unification and LangChain integration
- **Performance optimized** LangChain implementation with 60% memory reduction
- **API standardized** to v1.0 specification with unified `/wss/v1/audio` endpoint
- **Chatbot names**: Currently "부기" and "꼬기" in code, transitioning to "쫑이" and "아리"