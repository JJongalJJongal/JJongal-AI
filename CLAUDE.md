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

### WebSocket API Endpoints
- **Main Endpoint**: `/wss/v1/audio` - Unified audio processing with real-time voice cloning
- **Legacy Endpoints**: `/api/v1/wss/` and `/wss/v1/legacy/` - Backward compatibility

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
- **Backend**: FastAPI, SQLite, WebSockets
- **AI/ML**: OpenAI (GPT-4, DALL-E 3, Whisper), ElevenLabs, LangChain
- **Vector DB**: ChromaDB with OpenAI embeddings
- **Audio**: RNNoise for noise reduction, real-time voice cloning
- **Deployment**: Docker, Docker Compose, Nginx

### Data Flow
```
[Child Voice Input] → [WebSocket Audio Processing] → [쫑이 (ChatBot A): Story Collection + Voice Cloning]
                                                         ↓
[RAG System: Story Enhancement] → [아리 (ChatBot B): Complete Story Generation] → [Multimedia Output]
```

### Configuration
- **Environment Variables**: Defined in `.env` file (OpenAI, ElevenLabs API keys)
- **App Config**: `src/shared/configs/app.py` - Central configuration management
- **Prompts**: `src/shared/configs/prompts.py` - Age-specific conversation prompts

### Testing
- **Test Suite**: Use existing files in `tests/unit/` and `tests/integration/`
- **Manual Testing**: WebSocket test endpoints and demo commands available
- **Health Checks**: Built-in health monitoring at `/api/v1/health`

### Output Structure
```
output/
├── stories/           # Generated story content and metadata
├── temp/             # Temporary files (audio samples, images)
├── conversations/    # Saved conversation history
└── workflow_states/  # Story generation state management
```

### Important Notes
- **Voice Cloning**: Requires 5 audio samples before cloning activation
- **Binary Audio Transfer**: WebSocket audio endpoints expect binary data, not JSON
- **Age-Based Processing**: Different prompt strategies for 4-7 vs 8-9 age groups
- **Memory Management**: Persistent conversation storage with SQLite + LangChain
- **Resource Limits**: Docker containers configured with memory limits (8GB for AI services)

### Common Development Tasks (MODIFY EXISTING FILES ONLY)
- **Adding New Story Elements**: Modify `src/core/chatbots/chat_bot_a/chat_bot_a.py`
- **Updating Voice Processing**: Edit `src/core/voice/` components  
- **RAG System Changes**: Work with `src/data/vector_db/` modules
- **WebSocket Modifications**: Update `src/api/v1/endpoints/` handlers
- **Chatbot Personality Updates**: Edit prompts in `src/shared/configs/prompts.py`
- **Vector Database Management**: Use `scripts/data/manage_vector_db.py`

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