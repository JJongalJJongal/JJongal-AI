"""
JJongal-AI Test Configuration
Pytest configuration for async tests and fixtures
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()
    mock_client.images.generate = AsyncMock()
    return mock_client

@pytest.fixture  
def mock_elevenlabs_client():
    """Mock ElevenLabs client for testing"""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert = AsyncMock()
    mock_client.voices.get_all = AsyncMock()
    return mock_client

@pytest.fixture
def mock_progress_callback():
    """Mock progress callback for testing"""
    return AsyncMock()