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

### WebSocket API Endpoints
- **Main Endpoint**: `/wss/v1/audio` - Unified audio processing with real-time voice cloning
- **Legacy Endpoints**: `/api/v1/wss/` and `/wss/v1/legacy/` - Backward compatibility

### Key Features
1. **Real-time Voice Cloning**: Automatically collects and clones child's voice during conversation
2. **RAG System**: LangChain-based retrieval system using existing fairy tale knowledge
3. **Multimedia Generation**: DALL-E 3 images, ElevenLabs voice synthesis
4. **Age-Appropriate Content**: Specialized prompts for 4-7 and 8-9 age groups
5. **WebSocket Streaming**: Real-time audio processing with binary data transfer

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
- **App Config**: `shared/configs/app_config.py` - Central configuration management
- **Prompts**: `shared/configs/prompts_config.py` - Age-specific conversation prompts

### Testing
- **Test Suite**: `chatbot/tests/test_chatbot.py`
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