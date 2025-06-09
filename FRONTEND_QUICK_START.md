# 🚀 프론트엔드 개발자를 위한 꼬꼬북 API 빠른 시작

## 📋 목차
- [시작하기](#시작하기)
- [인증 설정](#인증-설정)
- [주요 기능 구현](#주요-기능-구현)
- [React 예제](#react-예제)
- [에러 처리](#에러-처리)
- [최적화 팁](#최적화-팁)

---

## 🎯 시작하기

### 개발 환경 설정
```bash
# 프로젝트 생성
npx create-react-app ccb-frontend
cd ccb-frontend

# 필요한 패키지 설치
npm install axios ws
```

### 기본 설정
```typescript
// src/config/api.ts
export const API_CONFIG = {
  baseUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000',
  timeout: 30000
};
```

---

## 🔐 인증 설정

### 1. 토큰 획득 및 저장
```typescript
// src/services/auth.ts
import axios from 'axios';
import { API_CONFIG } from '../config/api';

class AuthService {
  private tokenKey = 'ccb_auth_token';

  async getToken(): Promise<string> {
    // 저장된 토큰 확인
    const savedToken = localStorage.getItem(this.tokenKey);
    if (savedToken && this.isTokenValid(savedToken)) {
      return savedToken;
    }

    // 새 토큰 요청
    const response = await axios.get(`${API_CONFIG.baseUrl}/api/test-token`);
    const { token } = response.data;
    
    localStorage.setItem(this.tokenKey, token);
    return token;
  }

  private isTokenValid(token: string): boolean {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 > Date.now();
    } catch {
      return false;
    }
  }

  clearToken(): void {
    localStorage.removeItem(this.tokenKey);
  }
}

export const authService = new AuthService();
```

### 2. Axios 인터셉터 설정
```typescript
// src/services/api.ts
import axios from 'axios';
import { authService } from './auth';
import { API_CONFIG } from '../config/api';

const apiClient = axios.create({
  baseURL: API_CONFIG.baseUrl,
  timeout: API_CONFIG.timeout
});

// 요청 인터셉터
apiClient.interceptors.request.use(async (config) => {
  const token = await authService.getToken();
  config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// 응답 인터셉터
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      authService.clearToken();
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export { apiClient };
```

---

## 🎪 주요 기능 구현

### 1. 이야기 생성 서비스
```typescript
// src/services/story.ts
import { apiClient } from './api';

export interface ChildProfile {
  name: string;
  age: number;
  interests: string[];
  language_level: 'basic' | 'intermediate' | 'advanced';
  special_needs: string[];
}

export interface StoryCreationRequest {
  child_profile: ChildProfile;
  conversation_data?: any;
  story_preferences?: any;
  enable_multimedia: boolean;
}

class StoryService {
  async createStory(request: StoryCreationRequest) {
    const response = await apiClient.post('/api/v1/stories', request);
    return response.data;
  }

  async getStory(storyId: string) {
    const response = await apiClient.get(`/api/v1/stories/${storyId}`);
    return response.data;
  }

  async getStoryStatus(storyId: string) {
    const response = await apiClient.get(`/api/v1/stories/${storyId}/status`);
    return response.data;
  }

  async getStoryList(activeOnly = false) {
    const response = await apiClient.get(`/api/v1/stories?active_only=${activeOnly}`);
    return response.data;
  }

  async cancelStory(storyId: string) {
    const response = await apiClient.post(`/api/v1/stories/${storyId}/cancel`);
    return response.data;
  }
}

export const storyService = new StoryService();
```

### 2. WebSocket 오디오 서비스
```typescript
// src/services/websocket.ts
import { authService } from './auth';
import { API_CONFIG } from '../config/api';

export interface AudioMessage {
  type: 'audio_chunk' | 'conversation_end';
  data?: string;
  chunk_index?: number;
  is_final?: boolean;
}

export interface ServerMessage {
  type: 'transcription' | 'ai_response' | 'error';
  text?: string;
  audio_url?: string;
  confidence?: number;
  error_code?: string;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private messageHandlers: Map<string, (data: any) => void> = new Map();

  async connectAudio(params: {
    child_name: string;
    age: number;
    interests: string;
  }): Promise<void> {
    const token = await authService.getToken();
    const queryParams = new URLSearchParams({
      ...params,
      age: params.age.toString(),
      token
    });

    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${API_CONFIG.wsUrl}/ws/audio?${queryParams}`);

      this.ws.onopen = () => {
        console.log('WebSocket 연결됨');
        resolve();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket 에러:', error);
        reject(error);
      };

      this.ws.onmessage = (event) => {
        try {
          const message: ServerMessage = JSON.parse(event.data);
          const handler = this.messageHandlers.get(message.type);
          if (handler) {
            handler(message);
          }
        } catch (error) {
          console.error('메시지 파싱 에러:', error);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket 연결 종료');
        this.ws = null;
      };
    });
  }

  sendAudioChunk(audioData: ArrayBuffer, chunkIndex: number, isFinal: boolean): void {
    if (!this.ws) return;

    const base64Data = this.arrayBufferToBase64(audioData);
    const message: AudioMessage = {
      type: 'audio_chunk',
      data: base64Data,
      chunk_index: chunkIndex,
      is_final: isFinal
    };

    this.ws.send(JSON.stringify(message));
  }

  endConversation(): void {
    if (!this.ws) return;

    const message: AudioMessage = {
      type: 'conversation_end'
    };

    this.ws.send(JSON.stringify(message));
  }

  onMessage(type: string, handler: (data: any) => void): void {
    this.messageHandlers.set(type, handler);
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageHandlers.clear();
  }

  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}

export const webSocketService = new WebSocketService();
```

### 3. 오디오 녹음 서비스
```typescript
// src/services/audio.ts
class AudioRecordingService {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private stream: MediaStream | null = null;

  async startRecording(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });

      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: 'audio/webm'
      });

      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.start(1000); // 1초마다 데이터 전송
    } catch (error) {
      console.error('녹음 시작 실패:', error);
      throw error;
    }
  }

  stopRecording(): Promise<Blob> {
    return new Promise((resolve) => {
      if (!this.mediaRecorder) {
        resolve(new Blob());
        return;
      }

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        resolve(audioBlob);
      };

      this.mediaRecorder.stop();
      
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
      }
    });
  }

  isRecording(): boolean {
    return this.mediaRecorder?.state === 'recording';
  }
}

export const audioRecordingService = new AudioRecordingService();
```

---

## ⚛️ React 예제

### 1. 이야기 생성 컴포넌트
```tsx
// src/components/StoryCreator.tsx
import React, { useState } from 'react';
import { storyService, ChildProfile } from '../services/story';

const StoryCreator: React.FC = () => {
  const [childProfile, setChildProfile] = useState<ChildProfile>({
    name: '',
    age: 5,
    interests: [],
    language_level: 'basic',
    special_needs: []
  });
  const [loading, setLoading] = useState(false);
  const [storyId, setStoryId] = useState<string | null>(null);

  const handleCreateStory = async () => {
    if (!childProfile.name) {
      alert('아이 이름을 입력해주세요');
      return;
    }

    setLoading(true);
    try {
      const response = await storyService.createStory({
        child_profile: childProfile,
        enable_multimedia: true
      });

      if (response.success) {
        setStoryId(response.story_id);
        console.log('이야기 생성 시작:', response.story_id);
      }
    } catch (error) {
      console.error('이야기 생성 실패:', error);
      alert('이야기 생성에 실패했습니다');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="story-creator">
      <h2>새 이야기 만들기</h2>
      
      <div className="form-group">
        <label>아이 이름:</label>
        <input
          type="text"
          value={childProfile.name}
          onChange={(e) => setChildProfile({...childProfile, name: e.target.value})}
          placeholder="아이 이름을 입력하세요"
        />
      </div>

      <div className="form-group">
        <label>나이:</label>
        <select
          value={childProfile.age}
          onChange={(e) => setChildProfile({...childProfile, age: parseInt(e.target.value)})}
        >
          {[3,4,5,6,7,8,9,10,11,12].map(age => (
            <option key={age} value={age}>{age}세</option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>관심사:</label>
        <input
          type="text"
          placeholder="공주, 마법, 동물 (쉼표로 구분)"
          onChange={(e) => {
            const interests = e.target.value.split(',').map(s => s.trim()).filter(s => s);
            setChildProfile({...childProfile, interests});
          }}
        />
      </div>

      <button
        onClick={handleCreateStory}
        disabled={loading || !childProfile.name}
        className="create-button"
      >
        {loading ? '생성 중...' : '이야기 만들기'}
      </button>

      {storyId && (
        <div className="success-message">
          이야기 생성이 시작되었습니다! ID: {storyId}
        </div>
      )}
    </div>
  );
};

export default StoryCreator;
```

### 2. 음성 대화 컴포넌트
```tsx
// src/components/VoiceChat.tsx
import React, { useState, useEffect } from 'react';
import { webSocketService } from '../services/websocket';
import { audioRecordingService } from '../services/audio';

const VoiceChat: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<Array<{type: string, text: string}>>([]);
  const [childName, setChildName] = useState('');
  const [age, setAge] = useState(5);

  useEffect(() => {
    // WebSocket 메시지 핸들러 설정
    webSocketService.onMessage('transcription', (data) => {
      setMessages(prev => [...prev, {type: 'user', text: data.text}]);
    });

    webSocketService.onMessage('ai_response', (data) => {
      setMessages(prev => [...prev, {type: 'ai', text: data.text}]);
      
      // AI 응답 음성 재생
      if (data.audio_url) {
        const audio = new Audio(data.audio_url);
        audio.play();
      }
    });

    webSocketService.onMessage('error', (data) => {
      console.error('WebSocket 에러:', data.message);
      alert(`에러: ${data.message}`);
    });

    return () => {
      webSocketService.disconnect();
    };
  }, []);

  const handleConnect = async () => {
    if (!childName) {
      alert('아이 이름을 입력해주세요');
      return;
    }

    try {
      await webSocketService.connectAudio({
        child_name: childName,
        age: age,
        interests: '공주,마법,동물'
      });
      setIsConnected(true);
    } catch (error) {
      console.error('연결 실패:', error);
      alert('연결에 실패했습니다');
    }
  };

  const handleStartRecording = async () => {
    try {
      await audioRecordingService.startRecording();
      setIsRecording(true);
    } catch (error) {
      console.error('녹음 시작 실패:', error);
      alert('마이크 권한을 허용해주세요');
    }
  };

  const handleStopRecording = async () => {
    const audioBlob = await audioRecordingService.stopRecording();
    setIsRecording(false);

    // 오디오 데이터를 ArrayBuffer로 변환 후 전송
    const arrayBuffer = await audioBlob.arrayBuffer();
    webSocketService.sendAudioChunk(arrayBuffer, 1, true);
  };

  const handleDisconnect = () => {
    webSocketService.disconnect();
    setIsConnected(false);
    setMessages([]);
  };

  return (
    <div className="voice-chat">
      <h2>음성 대화</h2>

      {!isConnected ? (
        <div className="connection-form">
          <input
            type="text"
            placeholder="아이 이름"
            value={childName}
            onChange={(e) => setChildName(e.target.value)}
          />
          <select value={age} onChange={(e) => setAge(parseInt(e.target.value))}>
            {[4,5,6,7,8,9].map(a => <option key={a} value={a}>{a}세</option>)}
          </select>
          <button onClick={handleConnect}>연결하기</button>
        </div>
      ) : (
        <div className="chat-interface">
          <div className="messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.type}`}>
                <strong>{msg.type === 'user' ? childName : 'AI'}:</strong> {msg.text}
              </div>
            ))}
          </div>

          <div className="controls">
            <button
              onMouseDown={handleStartRecording}
              onMouseUp={handleStopRecording}
              onTouchStart={handleStartRecording}
              onTouchEnd={handleStopRecording}
              className={`record-button ${isRecording ? 'recording' : ''}`}
            >
              {isRecording ? '🔴 녹음 중...' : '🎤 말하기'}
            </button>
            
            <button onClick={handleDisconnect} className="disconnect-button">
              연결 종료
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceChat;
```

### 3. 이야기 상태 모니터링 컴포넌트
```tsx
// src/components/StoryStatus.tsx
import React, { useState, useEffect } from 'react';
import { storyService } from '../services/story';

interface Props {
  storyId: string;
}

const StoryStatus: React.FC<Props> = ({ storyId }) => {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await storyService.getStoryStatus(storyId);
        setStatus(response);
        
        // 완료되면 폴링 중지
        if (response.workflow_state === 'completed') {
          setLoading(false);
        }
      } catch (error) {
        console.error('상태 조회 실패:', error);
      }
    };

    // 3초마다 상태 확인
    const interval = setInterval(checkStatus, 3000);
    checkStatus(); // 즉시 한 번 실행

    return () => clearInterval(interval);
  }, [storyId]);

  const getProgressColor = (percentage: number) => {
    if (percentage < 30) return '#ff6b6b';
    if (percentage < 70) return '#feca57';
    return '#48cae4';
  };

  if (!status) return <div>상태 로딩 중...</div>;

  return (
    <div className="story-status">
      <h3>이야기 생성 상태</h3>
      
      <div className="status-info">
        <p><strong>ID:</strong> {status.story_id}</p>
        <p><strong>현재 단계:</strong> {status.current_stage}</p>
        <p><strong>상태:</strong> {status.workflow_state}</p>
      </div>

      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{
            width: `${status.progress_percentage}%`,
            backgroundColor: getProgressColor(status.progress_percentage)
          }}
        />
        <span className="progress-text">{status.progress_percentage.toFixed(1)}%</span>
      </div>

      {status.errors?.length > 0 && (
        <div className="errors">
          <h4>오류:</h4>
          {status.errors.map((error: string, idx: number) => (
            <p key={idx} className="error">{error}</p>
          ))}
        </div>
      )}

      {status.workflow_state === 'completed' && (
        <div className="completion-message">
          🎉 이야기 생성이 완료되었습니다!
          <button onClick={() => window.open(`/story/${storyId}`, '_blank')}>
            이야기 보기
          </button>
        </div>
      )}
    </div>
  );
};

export default StoryStatus;
```

---

## ⚠️ 에러 처리

### 전역 에러 처리기
```typescript
// src/utils/errorHandler.ts
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public errorCode?: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export const handleApiError = (error: any): string => {
  if (error.response?.data) {
    const { message, error_code } = error.response.data;
    return `${message} (${error_code})`;
  }
  
  if (error.message) {
    return error.message;
  }
  
  return '알 수 없는 오류가 발생했습니다';
};

export const isNetworkError = (error: any): boolean => {
  return !error.response && error.request;
};
```

### React 에러 바운더리
```tsx
// src/components/ErrorBoundary.tsx
import React, { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>문제가 발생했습니다</h2>
          <p>페이지를 새로고침해주세요</p>
          <button onClick={() => window.location.reload()}>
            새로고침
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

---

## 🚀 최적화 팁

### 1. API 응답 캐싱
```typescript
// src/utils/cache.ts
class ResponseCache {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private ttl = 5 * 60 * 1000; // 5분

  set(key: string, data: any): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }
}

export const responseCache = new ResponseCache();
```

### 2. WebSocket 재연결 로직
```typescript
// src/services/websocket.ts (추가)
class WebSocketService {
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  private async reconnect(params: any): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      throw new Error('최대 재연결 시도 횟수 초과');
    }

    this.reconnectAttempts++;
    await new Promise(resolve => setTimeout(resolve, this.reconnectDelay));
    
    try {
      await this.connectAudio(params);
      this.reconnectAttempts = 0; // 성공 시 리셋
    } catch (error) {
      return this.reconnect(params);
    }
  }
}
```

### 3. 메모리 최적화
```typescript
// src/hooks/useCleanup.ts
import { useEffect } from 'react';

export const useCleanup = (cleanup: () => void) => {
  useEffect(() => {
    return cleanup;
  }, [cleanup]);
};

// 사용 예시
const MyComponent = () => {
  useCleanup(() => {
    webSocketService.disconnect();
    audioRecordingService.stopRecording();
  });
  
  // 컴포넌트 로직...
};
```

---

## 📱 모바일 최적화

### 터치 이벤트 처리
```typescript
// src/components/MobileVoiceButton.tsx
const MobileVoiceButton: React.FC = () => {
  const [isPressed, setIsPressed] = useState(false);

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    setIsPressed(true);
    audioRecordingService.startRecording();
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    e.preventDefault();
    setIsPressed(false);
    audioRecordingService.stopRecording();
  };

  return (
    <button
      className={`voice-button ${isPressed ? 'pressed' : ''}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onContextMenu={(e) => e.preventDefault()} // 길게 눌러도 메뉴 안 뜨게
    >
      🎤 {isPressed ? '녹음 중...' : '말하기'}
    </button>
  );
};
```

---

## 📱 **프론트엔드 빠른 시작 가이드**

이 가이드는 꼬꼬북 AI의 WebSocket API를 사용하여 프론트엔드를 빠르게 구축하는 방법을 설명합니다.

### 🎯 **Binary 데이터 처리 (개선된 방식)**

서버에서 오디오와 이미지를 WebSocket binary로 전송합니다. 청킹을 통해 순서를 보장합니다.

```typescript
// src/services/websocket-binary.ts
export interface AudioMetadata {
  type: 'audio_metadata';
  size: number;
  size_mb: number;
  format: string;
  chunks_total: number;
  chunk_size: number;
  sequence_id: number;
}

export interface ChunkHeader {
  type: 'audio_chunk_header' | 'story_file_chunk_header';
  sequence_id: number;
  chunk_index: number;
  total_chunks: number;
  chunk_size: number;
  is_final: boolean;
}

class BinaryDataReceiver {
  private audioChunks: Map<number, Uint8Array[]> = new Map();
  private onAudioComplete: (audioBlob: Blob, metadata: AudioMetadata) => void;
  
  constructor(onAudioComplete: (audioBlob: Blob, metadata: AudioMetadata) => void) {
    this.onAudioComplete = onAudioComplete;
  }

  handleMessage(event: MessageEvent) {
    if (typeof event.data === 'string') {
      // JSON 메시지 처리
      const message = JSON.parse(event.data);
      
      switch (message.type) {
        case 'audio_metadata':
          this.handleAudioMetadata(message as AudioMetadata);
          break;
        case 'audio_chunk_header':
          this.handleChunkHeader(message as ChunkHeader);
          break;
        case 'audio_transfer_complete':
          this.handleTransferComplete(message);
          break;
      }
    } else if (event.data instanceof ArrayBuffer) {
      // Binary 데이터 처리
      this.handleBinaryData(new Uint8Array(event.data));
    }
  }

  private handleAudioMetadata(metadata: AudioMetadata) {
    console.log(`🎵 오디오 수신 준비: ${metadata.size_mb}MB, ${metadata.chunks_total} 청크`);
    // 청크 배열 초기화
    this.audioChunks.set(metadata.sequence_id, []);
  }

  private currentChunkHeader: ChunkHeader | null = null;

  private handleChunkHeader(header: ChunkHeader) {
    this.currentChunkHeader = header;
    console.log(`📦 청크 ${header.chunk_index + 1}/${header.total_chunks} 수신 준비`);
  }

  private handleBinaryData(data: Uint8Array) {
    if (!this.currentChunkHeader) {
      console.warn('청크 헤더 없이 binary 데이터 수신');
      return;
    }

    const header = this.currentChunkHeader;
    const chunks = this.audioChunks.get(header.sequence_id) || [];
    
    // 청크 추가
    chunks[header.chunk_index] = data;
    this.audioChunks.set(header.sequence_id, chunks);
    
    console.log(`✅ 청크 ${header.chunk_index + 1}/${header.total_chunks} 수신 완료`);
    
    // 마지막 청크인 경우 조립
    if (header.is_final) {
      this.assembleAudio(header.sequence_id);
    }
    
    this.currentChunkHeader = null;
  }

  private handleTransferComplete(message: any) {
    console.log(`🎉 오디오 전송 완료: ${message.total_chunks} 청크, ${message.total_size} bytes`);
  }

  private assembleAudio(sequenceId: number) {
    const chunks = this.audioChunks.get(sequenceId);
    if (!chunks) return;

    // 청크들을 순서대로 조립
    let totalSize = 0;
    chunks.forEach(chunk => totalSize += chunk.length);
    
    const assembled = new Uint8Array(totalSize);
    let offset = 0;
    
    chunks.forEach(chunk => {
      assembled.set(chunk, offset);
      offset += chunk.length;
    });

    // Blob 생성하여 콜백 호출
    const audioBlob = new Blob([assembled], { type: 'audio/mpeg' });
    console.log(`🔧 오디오 조립 완료: ${audioBlob.size} bytes`);
    
    // 메타데이터와 함께 콜백 호출 (메타데이터는 별도로 저장해야 함)
    this.onAudioComplete(audioBlob, { 
      type: 'audio_metadata',
      size: audioBlob.size,
      size_mb: audioBlob.size / (1024 * 1024),
      format: 'mp3',
      chunks_total: chunks.length,
      chunk_size: 1024 * 1024,
      sequence_id: sequenceId
    });
    
    // 정리
    this.audioChunks.delete(sequenceId);
  }
}
```

### 🎙️ **음성 대화용 WebSocket 클라이언트**

```typescript
// src/services/voice-chat.ts
import { BinaryDataReceiver } from './websocket-binary';

class VoiceChatService {
  private ws: WebSocket | null = null;
  private binaryReceiver: BinaryDataReceiver;
  private audioQueue: HTMLAudioElement[] = [];
  private isPlaying = false;

  constructor() {
    this.binaryReceiver = new BinaryDataReceiver((audioBlob, metadata) => {
      this.handleReceivedAudio(audioBlob, metadata);
    });
  }

  async connect(childName: string, age: number, interests: string) {
    const url = `ws://localhost:8001/ws/audio?${new URLSearchParams({
      child_name: childName,
      age: age.toString(),
      interests: interests,
      token: 'development_token'
    })}`;

    this.ws = new WebSocket(url);
    
    this.ws.onopen = () => {
      console.log('🔗 음성 대화 연결 성공');
    };

    this.ws.onmessage = (event) => {
      // Binary 데이터 처리는 BinaryDataReceiver에 위임
      this.binaryReceiver.handleMessage(event);
      
      // JSON 메시지 처리
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);
        this.handleTextMessage(message);
      }
    };

    this.ws.onerror = (error) => {
      console.error('❌ WebSocket 오류:', error);
    };

    this.ws.onclose = () => {
      console.log('🔌 연결 종료됨');
    };
  }

  private handleTextMessage(message: any) {
    switch (message.type) {
      case 'conversation_response':
        console.log('💬 AI 응답:', message.text);
        console.log('🎤 사용자 발언:', message.user_text);
        console.log('🔊 오디오 방식:', message.audio_method);
        
        // base64 fallback 처리
        if (message.audio_method === 'base64_fallback' && message.audio) {
          this.playBase64Audio(message.audio);
        }
        break;
        
      case 'audio_metadata':
        console.log(`🎵 오디오 수신 시작: ${message.size_mb}MB`);
        break;
        
      case 'error':
        console.error('❌ 서버 오류:', message.error_message);
        break;
    }
  }

  private handleReceivedAudio(audioBlob: Blob, metadata: any) {
    console.log(`🎵 오디오 수신 완료: ${metadata.size_mb}MB`);
    
    // Blob을 Audio 객체로 변환
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    
    // 재생 큐에 추가
    this.audioQueue.push(audio);
    
    // 큐가 비어있으면 즉시 재생
    if (!this.isPlaying) {
      this.playNextAudio();
    }
  }

  private playBase64Audio(base64Data: string) {
    const audio = new Audio(`data:audio/mpeg;base64,${base64Data}`);
    this.audioQueue.push(audio);
    
    if (!this.isPlaying) {
      this.playNextAudio();
    }
  }

  private async playNextAudio() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }

    this.isPlaying = true;
    const audio = this.audioQueue.shift()!;
    
    return new Promise<void>((resolve) => {
      audio.onended = () => {
        URL.revokeObjectURL(audio.src); // 메모리 정리
        resolve();
        this.playNextAudio(); // 다음 오디오 재생
      };
      
      audio.onerror = (error) => {
        console.error('🔊 오디오 재생 오류:', error);
        resolve();
        this.playNextAudio();
      };
      
      audio.play().catch(error => {
        console.error('🔊 오디오 재생 실패:', error);
        resolve();
        this.playNextAudio();
      });
    });
  }

  sendAudioData(audioData: ArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // 바이너리 오디오 데이터 전송
      this.ws.send(audioData);
    }
  }

  sendAudioEnd() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // 오디오 종료 신호 전송
      this.ws.send(JSON.stringify({ type: 'audio_end' }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    // 재생 중인 오디오들 정리
    this.audioQueue.forEach(audio => {
      audio.pause();
      URL.revokeObjectURL(audio.src);
    });
    this.audioQueue = [];
    this.isPlaying = false;
  }
}

export const voiceChatService = new VoiceChatService();
```

### 📚 **스토리 생성용 WebSocket 클라이언트**

```typescript
// src/services/story-generation.ts
class StoryGenerationService {
  private ws: WebSocket | null = null;
  private receivedFiles: Map<string, { blob: Blob, metadata: any }> = new Map();
  private onStoryProgress: (progress: number, message: string) => void = () => {};
  private onFileReceived: (fileBlob: Blob, metadata: any) => void = () => {};
  private onStoryComplete: (storyData: any) => void = () => {};

  async connectStoryGeneration(childName: string, age: number, interests: string) {
    const url = `ws://localhost:8001/ws/story_generation?${new URLSearchParams({
      child_name: childName,
      age: age.toString(),
      interests: interests,
      token: 'development_token'
    })}`;

    this.ws = new WebSocket(url);
    
    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data);
        this.handleStoryMessage(message);
      } else if (event.data instanceof ArrayBuffer) {
        this.handleStoryBinaryData(new Uint8Array(event.data));
      }
    };
  }

  private currentFileMetadata: any = null;
  private currentChunks: Uint8Array[] = [];

  private handleStoryMessage(message: any) {
    switch (message.type) {
      case 'story_metadata':
        console.log(`📖 스토리 시작: ${message.title}`);
        this.onStoryProgress(20, `${message.title} 생성 시작...`);
        break;
        
      case 'story_file_metadata':
        console.log(`📁 파일 수신 시작: ${message.file_type} ch${message.chapter}`);
        this.currentFileMetadata = message;
        this.currentChunks = [];
        
        const progress = (message.sequence_index / message.sequence_total) * 80 + 20;
        this.onStoryProgress(progress, `${message.file_type} 파일 수신 중...`);
        break;
        
      case 'story_file_chunk_header':
        console.log(`📦 청크 ${message.chunk_index + 1}/${message.total_chunks} 수신 준비`);
        break;
        
      case 'story_file_complete':
        this.assembleStoryFile();
        break;
        
      case 'story_transfer_complete':
        console.log(`🎉 스토리 완성: ${message.title}`);
        this.onStoryProgress(100, '스토리 완성!');
        this.onStoryComplete(message);
        break;
        
      case 'error':
        console.error('❌ 스토리 생성 오류:', message.error_message);
        break;
    }
  }

  private handleStoryBinaryData(data: Uint8Array) {
    if (!this.currentFileMetadata) {
      console.warn('파일 메타데이터 없이 binary 데이터 수신');
      return;
    }

    // 청크 추가
    this.currentChunks.push(data);
    console.log(`✅ 청크 수신: ${data.length} bytes`);
  }

  private assembleStoryFile() {
    if (!this.currentFileMetadata || this.currentChunks.length === 0) return;

    // 청크들 조립
    let totalSize = 0;
    this.currentChunks.forEach(chunk => totalSize += chunk.length);
    
    const assembled = new Uint8Array(totalSize);
    let offset = 0;
    
    this.currentChunks.forEach(chunk => {
      assembled.set(chunk, offset);
      offset += chunk.length;
    });

    // 파일 타입에 따른 MIME 타입 결정
    const mimeType = this.currentFileMetadata.file_type === 'image' ? 'image/png' : 'audio/mpeg';
    const fileBlob = new Blob([assembled], { type: mimeType });
    
    console.log(`🔧 파일 조립 완료: ${this.currentFileMetadata.file_type} ${fileBlob.size} bytes`);
    
    // 파일 저장 및 콜백 호출
    const fileKey = `${this.currentFileMetadata.sequence_index}_${this.currentFileMetadata.file_type}_ch${this.currentFileMetadata.chapter}`;
    this.receivedFiles.set(fileKey, { blob: fileBlob, metadata: this.currentFileMetadata });
    
    this.onFileReceived(fileBlob, this.currentFileMetadata);
    
    // 정리
    this.currentFileMetadata = null;
    this.currentChunks = [];
  }

  setProgressCallback(callback: (progress: number, message: string) => void) {
    this.onStoryProgress = callback;
  }

  setFileReceivedCallback(callback: (fileBlob: Blob, metadata: any) => void) {
    this.onFileReceived = callback;
  }

  setStoryCompleteCallback(callback: (storyData: any) => void) {
    this.onStoryComplete = callback;
  }

  getReceivedFiles() {
    return this.receivedFiles;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.receivedFiles.clear();
  }
}

export const storyGenerationService = new StoryGenerationService();
```

### 🎮 **React 컴포넌트 예제**

```tsx
// src/components/VoiceChat.tsx
import React, { useState, useEffect } from 'react';
import { voiceChatService } from '../services/voice-chat';

const VoiceChat: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);

  const handleConnect = async () => {
    await voiceChatService.connect('테스트', 7, '공주,마법');
    setIsConnected(true);
  };

  const handleStartRecording = async () => {
    // 마이크 녹음 시작
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        event.data.arrayBuffer().then(buffer => {
          voiceChatService.sendAudioData(buffer);
        });
      }
    };
    
    mediaRecorder.start(1000); // 1초마다 데이터 전송
    setIsRecording(true);
    
    // 3초 후 자동 중지 (테스트용)
    setTimeout(() => {
      mediaRecorder.stop();
      voiceChatService.sendAudioEnd();
      setIsRecording(false);
    }, 3000);
  };

  return (
    <div className="voice-chat">
      <h2>🎤 음성 대화</h2>
      
      {!isConnected ? (
        <button onClick={handleConnect}>연결하기</button>
      ) : (
        <div>
          <button 
            onClick={handleStartRecording} 
            disabled={isRecording}
            className={isRecording ? 'recording' : ''}
          >
            {isRecording ? '🔴 녹음 중...' : '🎤 말하기'}
          </button>
          
          <div className="messages">
            {messages.map((msg, idx) => (
              <div key={idx}>{msg}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VoiceChat;
```

이제 프론트엔드에서 WebSocket binary 데이터를 순서대로 받아서 처리할 수 있습니다! 🎉

**주요 특징:**
- ✅ **순서 보장**: 청크 헤더 → binary 데이터 순서로 전송
- ✅ **대용량 지원**: 1MB 청크로 분할하여 안정적 전송  
- ✅ **자동 조립**: 프론트에서 청크들을 자동으로 조립
- ✅ **재생 큐**: 오디오들이 순서대로 재생됨
- ✅ **메모리 관리**: 사용 후 Blob URL 자동 정리

