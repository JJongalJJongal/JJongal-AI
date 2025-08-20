# CLAUDE.md - JJongal-AI Project Guidelines

This file provides **STRICT RULES AND GUIDANCE** for Claude Code when working with this repository.

## 🚫 CRITICAL RULES - NEVER BREAK THESE

### 1. **NEVER CREATE NEW FILES WITHOUT EXPLICIT PERMISSION**
- ❌ **DO NOT** create new Python files, test files, or any other files
- ❌ **DO NOT** create new folders or directories  
- ✅ **ONLY MODIFY** existing files that are already in the project structure
- ✅ **ASK PERMISSION** before creating any new file

### 2. **USE EXISTING STRUCTURE ONLY**
- The project structure is already complete and well-organized
- All necessary files, folders, and modules already exist
- Your job is to **IMPROVE EXISTING FILES**, not create new ones
- Work within the established architecture

### 3. **EXISTING LOCATIONS FOR EVERYTHING**

#### Tests (DO NOT CREATE NEW TEST FILES):
- ✅ Use: `tests/unit/test_chatbot_a.py`
- ✅ Use: `tests/unit/test_chatbot_b.py` 
- ✅ Use: `tests/integration/test_api.py`
- ✅ Use: `tests/integration/test_workflow.py`
- ❌ **DO NOT** create `test_modern_chatbots.py` or any new test files

#### Core Chatbots (MODIFY THESE):
- ✅ Use: `src/core/chatbots/chat_bot_a/chat_bot_a.py`
- ✅ Use: `src/core/chatbots/chat_bot_b/chat_bot_b.py`
- ✅ Use: `src/core/chatbots/collaboration/jjong_ari_collaborator.py`

#### Chains (MODIFY THESE):
- ✅ Use: `src/core/chatbots/chat_bot_a/chains/conversation_chain.py`
- ✅ Use: `src/core/chatbots/chat_bot_a/chains/rag_chain.py`

#### Documentation (USE EXISTING):
- ✅ Use: `docs/development/` for technical docs
- ✅ Use: `docs/api/` for API documentation  
- ✅ Use: `README.md` for project overview

### 4. **MODIFICATION APPROACH**
- **Enhance existing functions** rather than replacing them
- **Add features to existing classes** rather than creating new ones
- **Maintain backward compatibility** always
- **Keep existing import paths** working

### 5. **WHEN IN DOUBT**
- **ASK** "Should I modify existing file X or create a new file?"
- **DEFAULT** to modifying existing files
- **NEVER ASSUME** you need a new file

## 📋 Project Overview

CCB_AI is an EduTech project called "쫑알쫑알" (JjongAlJjongAl/Chattering) that creates interactive fairy tale books from children's imagination using AI. The system consists of two main AI chatbots working together:

- **ChatBot A (쫑이/Jjongi)**: Voice-interactive story collection bot that gathers story elements from children through real-time conversation
- **ChatBot B (아리/Ari)**: Story completion bot that generates complete multimedia fairy tales with text, images, and voice

## 🛠️ Development Commands

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

# Run tests (USE EXISTING TEST FILES ONLY)
python -m pytest tests/ -v

# Direct server start (development)
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
# Use existing scripts
python scripts/data/populate_vector_db.py
python scripts/data/manage_vector_db.py
```

## 🏗️ Architecture Overview

### Core Components
- **FastAPI Application** (`app.py`): Main server with WebSocket endpoints
- **ChatBot A (쫑이)** (`src/core/chatbots/chat_bot_a/`): Story collection via voice interaction
- **ChatBot B (아리)** (`src/core/chatbots/chat_bot_b/`): Complete story generation with multimedia
- **Vector Database** (`src/data/vector_db/`): LangChain + ChromaDB for RAG system
- **Workflow System** (`src/core/workflow/`): Orchestrates the complete story generation pipeline

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

## 🛠️ Development Commands

### Docker-based Development
```bash
# Build and run development environment
make dev                    # Start development environment with live reload
make build                  # Build Docker images
make run                    # Start production environment
make stop                   # Stop all services
```

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (USE EXISTING TEST FILES)
python -m pytest tests/ -v

# Direct server start (development)
python app.py
```

### Vector Database Management
```bash
# Use existing scripts
python scripts/data/populate_vector_db.py
python scripts/data/manage_vector_db.py
```

## 📁 File Structure Rules

```
JJongal-AI/
├── src/core/chatbots/           # MODIFY THESE FOR CHATBOT CHANGES
│   ├── chat_bot_a/              # ChatBot A (쫑이)
│   ├── chat_bot_b/              # ChatBot B (아리)  
│   └── collaboration/           # Cooperation between A & B
├── src/data/vector_db/          # MODIFY FOR RAG CHANGES
├── tests/                       # USE EXISTING TEST FILES ONLY
├── docs/                        # USE EXISTING DOCS STRUCTURE
└── config/                      # Configuration files
```

## ⚠️ REMEMBER

**Your role is to IMPROVE what exists, not CREATE what doesn't.**

If you feel tempted to create a new file, STOP and ask yourself:
1. "Is there already a file that does something similar?"  
2. "Can I add this functionality to an existing file?"
3. "Did the user explicitly ask for a new file?"

99% of the time, the answer is to modify an existing file.

---

### WebSocket API Endpoints
- **Main Endpoint**: `/wss/v1/audio` - Unified audio processing with real-time voice cloning
- **Legacy Endpoints**: `/api/v1/wss/` and `/wss/v1/legacy/` - Backward compatibility

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
- **Environment Variables**: Defined in `.env` file (OpenAI, ElevenLabs API keys)
- **App Config**: `shared/configs/app_config.py` - Central configuration management
- **Prompts**: `shared/configs/prompts_config.py` - Age-specific conversation prompts

### Testing
- **Test Suite**: Use existing files in `tests/unit/` and `tests/integration/`
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
- **Adding New Story Elements**: Modify `chatbot/models/chat_bot_a/core/story_engine.py`
- **Updating Voice Processing**: Edit `chatbot/models/voice_ws/` components  
- **RAG System Changes**: Work with `chatbot/data/vector_db/` modules
- **WebSocket Modifications**: Update `chatbot/api/v1/ws/` handlers
- **Chatbot Personality Updates**: Edit prompts in `shared/configs/prompts_config.py`
- **Vector Database Management**: Use `chatbot/data/vector_db/manage_vector_db.py`

### Project Evolution Notes
- **Service rebranded** from "꼬꼬북" to "쫑알쫑알" (legacy references may still exist in some files)
- **Recent major refactoring** focused on WebSocket unification and LangChain integration
- **Performance optimized** LangChain implementation with 60% memory reduction
- **API standardized** to v1.0 specification with unified `/wss/v1/audio` endpoint
- **Chatbot names**: Currently "부기" and "꼬기" in code, transitioning to "쫑이" and "아리"
- **2025 LangChain Best Practices Applied**: LCEL, modern RAG, simplified architecture

## ⚠️ REMEMBER

**Your role is to IMPROVE what exists, not CREATE what doesn't.**

If you feel tempted to create a new file, STOP and ask yourself:
1. "Is there already a file that does something similar?"  
2. "Can I add this functionality to an existing file?"
3. "Did the user explicitly ask for a new file?"

99% of the time, the answer is to modify an existing file.

**Follow these rules strictly. The project structure is intentional and complete.**