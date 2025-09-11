"""
Authentication Processor for Voice WebSocket Connections

Handles JWT token validation, user authentication, and security
for voice-based WebSocket connections in JJongal-AI.
"""

import asyncio
import time
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError

from src.shared.utils.logging import get_module_logger
from src.shared.configs.app import get_env_vars

logger = get_module_logger(__name__)

@dataclass
class AuthResult:
    """Authentication result"""
    is_valid: bool
    user_id: Optional[str] = None
    child_profile: Optional[Dict[str, Any]] = None
    permissions: Optional[Dict[str, bool]] = None
    error: Optional[str] = None
    expires_at: Optional[datetime] = None

class VoiceAuthProcessor:
    """
    Processes authentication for voice WebSocket connections
    
    Features:
    - JWT token validation
    - Child profile extraction
    - Permission checking
    - Session management
    - Rate limiting
    """
    
    def __init__(self):
        """Initialize authentication processor"""
        env_vars = get_env_vars()
        self.jwt_secret = env_vars.get("JWT_SECRET_KEY", "default-secret")
        self.jwt_algorithm = "HS256"
        self.token_expiry_hours = 24
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # Security settings
        self.max_requests_per_minute = 60
        self.max_sessions_per_user = 3
        self.session_timeout_minutes = 30
        
        logger.info("Voice Authentication Processor initialized")
    
    async def authenticate_websocket_connection(
        self,
        token: str,
        websocket_id: str,
        client_ip: str = None
    ) -> AuthResult:
        """
        Authenticate WebSocket connection with JWT token
        
        Args:
            token: JWT authentication token
            websocket_id: WebSocket connection identifier
            client_ip: Client IP address for rate limiting
            
        Returns:
            AuthResult: Authentication result
        """
        try:
            # Rate limiting check
            if client_ip and not self._check_rate_limit(client_ip):
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return AuthResult(
                    is_valid=False,
                    error="Rate limit exceeded. Please try again later."
                )
            
            # Validate JWT token
            auth_data = await self._validate_jwt_token(token)
            if not auth_data:
                logger.warning("Invalid JWT token provided")
                return AuthResult(
                    is_valid=False,
                    error="Invalid authentication token"
                )
            
            user_id = auth_data.get("user_id")
            if not user_id:
                logger.warning("Missing user_id in token")
                return AuthResult(
                    is_valid=False,
                    error="Invalid token format"
                )
            
            # Check session limits
            if not self._check_session_limits(user_id):
                logger.warning(f"Session limit exceeded for user: {user_id}")
                return AuthResult(
                    is_valid=False,
                    error="Maximum sessions exceeded"
                )
            
            # Extract child profile and permissions
            child_profile = auth_data.get("child_profile", {})
            permissions = auth_data.get("permissions", {})
            expires_at = datetime.fromtimestamp(auth_data.get("exp", time.time()))
            
            # Create session
            await self._create_session(websocket_id, user_id, child_profile, client_ip)
            
            logger.info(f"Authentication successful for user: {user_id}")
            
            return AuthResult(
                is_valid=True,
                user_id=user_id,
                child_profile=child_profile,
                permissions=permissions,
                expires_at=expires_at
            )
            
        except ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return AuthResult(
                is_valid=False,
                error="Token has expired. Please login again."
            )
        except InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return AuthResult(
                is_valid=False,
                error="Invalid authentication token"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthResult(
                is_valid=False,
                error="Authentication failed"
            )
    
    async def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and extract payload
        
        Args:
            token: JWT token to validate
            
        Returns:
            Optional[Dict]: Token payload if valid
        """
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check required fields
            required_fields = ["user_id", "exp", "iat"]
            if not all(field in payload for field in required_fields):
                logger.warning("JWT token missing required fields")
                return None
            
            # Check if token is not expired (additional check)
            if payload.get("exp", 0) < time.time():
                logger.warning("JWT token is expired")
                return None
            
            return payload
            
        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client IP is within rate limits
        
        Args:
            client_ip: Client IP address
            
        Returns:
            bool: True if within limits
        """
        current_time = time.time()
        minute_window = int(current_time // 60)
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {}
        
        ip_limits = self.rate_limits[client_ip]
        
        # Clean old windows
        old_windows = [w for w in ip_limits.keys() if w < minute_window - 5]
        for old_window in old_windows:
            del ip_limits[old_window]
        
        # Check current window
        current_requests = ip_limits.get(minute_window, 0)
        if current_requests >= self.max_requests_per_minute:
            return False
        
        # Increment counter
        ip_limits[minute_window] = current_requests + 1
        return True
    
    def _check_session_limits(self, user_id: str) -> bool:
        """
        Check if user is within session limits
        
        Args:
            user_id: User identifier
            
        Returns:
            bool: True if within limits
        """
        # Clean expired sessions
        self._cleanup_expired_sessions()
        
        # Count active sessions for user
        user_sessions = sum(
            1 for session in self.active_sessions.values()
            if session.get("user_id") == user_id
        )
        
        return user_sessions < self.max_sessions_per_user
    
    async def _create_session(
        self,
        websocket_id: str,
        user_id: str,
        child_profile: Dict[str, Any],
        client_ip: str = None
    ):
        """
        Create new authentication session
        
        Args:
            websocket_id: WebSocket connection ID
            user_id: User identifier
            child_profile: Child profile information
            client_ip: Client IP address
        """
        session_data = {
            "websocket_id": websocket_id,
            "user_id": user_id,
            "child_profile": child_profile,
            "client_ip": client_ip,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "is_active": True
        }
        
        self.active_sessions[websocket_id] = session_data
        logger.debug(f"Created session for websocket: {websocket_id}")
    
    async def update_session_activity(self, websocket_id: str) -> bool:
        """
        Update session last activity timestamp
        
        Args:
            websocket_id: WebSocket connection ID
            
        Returns:
            bool: Success status
        """
        if websocket_id in self.active_sessions:
            self.active_sessions[websocket_id]["last_activity"] = datetime.now()
            return True
        return False
    
    async def close_session(self, websocket_id: str) -> bool:
        """
        Close authentication session
        
        Args:
            websocket_id: WebSocket connection ID
            
        Returns:
            bool: Success status
        """
        if websocket_id in self.active_sessions:
            session = self.active_sessions[websocket_id]
            session["is_active"] = False
            session["closed_at"] = datetime.now()
            
            # Remove from active sessions after a delay for logging
            await asyncio.sleep(0.1)  # Allow for any final logging
            del self.active_sessions[websocket_id]
            
            logger.info(f"Closed session for websocket: {websocket_id}")
            return True
        return False
    
    def get_session_info(self, websocket_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information
        
        Args:
            websocket_id: WebSocket connection ID
            
        Returns:
            Optional[Dict]: Session information
        """
        session = self.active_sessions.get(websocket_id)
        if not session:
            return None
        
        return {
            "user_id": session["user_id"],
            "child_profile": session["child_profile"],
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "duration_minutes": (
                datetime.now() - session["created_at"]
            ).total_seconds() / 60,
            "is_active": session["is_active"]
        }
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        timeout_threshold = current_time - timedelta(minutes=self.session_timeout_minutes)
        
        expired_sessions = [
            websocket_id for websocket_id, session in self.active_sessions.items()
            if session["last_activity"] < timeout_threshold
        ]
        
        for websocket_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {websocket_id}")
            del self.active_sessions[websocket_id]
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        self._cleanup_expired_sessions()
        return len(self.active_sessions)
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            List[Dict]: List of user sessions
        """
        self._cleanup_expired_sessions()
        
        user_sessions = []
        for websocket_id, session in self.active_sessions.items():
            if session["user_id"] == user_id:
                user_sessions.append({
                    "websocket_id": websocket_id,
                    "created_at": session["created_at"],
                    "last_activity": session["last_activity"],
                    "client_ip": session.get("client_ip"),
                    "is_active": session["is_active"]
                })
        
        return user_sessions
    
    async def verify_session_permissions(
        self,
        websocket_id: str,
        required_permission: str
    ) -> bool:
        """
        Verify session has required permission
        
        Args:
            websocket_id: WebSocket connection ID
            required_permission: Permission to check
            
        Returns:
            bool: True if permission granted
        """
        session = self.active_sessions.get(websocket_id)
        if not session:
            return False
        
        # Update last activity
        await self.update_session_activity(websocket_id)
        
        # Check child profile for permissions
        child_profile = session.get("child_profile", {})
        permissions = child_profile.get("permissions", {})
        
        # Default permissions for voice features
        default_permissions = {
            "voice_interaction": True,
            "voice_cloning": True,
            "story_generation": True,
            "audio_recording": True
        }
        
        return permissions.get(required_permission, 
                             default_permissions.get(required_permission, False))

# Global auth processor instance
_auth_processor = None

def get_auth_processor() -> VoiceAuthProcessor:
    """Get global authentication processor instance"""
    global _auth_processor
    if _auth_processor is None:
        _auth_processor = VoiceAuthProcessor()
    return _auth_processor