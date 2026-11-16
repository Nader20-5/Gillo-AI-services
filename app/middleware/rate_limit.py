import time
from typing import Dict, List
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Max request body size: 2 Megabytes
MAX_REQUEST_SIZE_BYTES = 2 * 1024 * 1024

# In-memory sliding window rate limiter: max 60 requests per 60 seconds per user/IP
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_REQUESTS_PER_WINDOW = 60

_request_timestamps: Dict[str, List[float]] = {}


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > MAX_REQUEST_SIZE_BYTES:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "Request payload exceeds the maximum allowed limit of 2MB."},
                    )
            except ValueError:
                pass
        return await call_next(request)


def check_llm_rate_limit(request: Request):
    """
    Dependency for LLM-backed endpoints enforcing sliding-window rate limiting.
    Security invariant: Limits are keyed on verified JWT sub claims or client IP.
    Unverified or forged tokens never consume another user's rate-limiting bucket.
    Raises 429 Too Many Requests if threshold is exceeded.
    """
    client_ip = request.client.host if request.client else "unknown"
    verified_sub = None

    auth = request.headers.get("authorization")
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ")[1]
        try:
            from jose import jwt
            from app.config import settings
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_signature": True, "verify_exp": True}
            )
            verified_sub = payload.get("sub") or payload.get("user_id")
        except Exception:
            # Signature invalid, expired, or malformed: do NOT trust claims!
            verified_sub = None
    elif request.headers.get("x-mock-user-id"):
        from app.config import settings
        if settings.ALLOW_MOCK_AUTH and settings.ENVIRONMENT.lower() != "production":
            verified_sub = request.headers.get("x-mock-user-id")

    limiter_key = f"user:{verified_sub}" if verified_sub else f"ip:{client_ip}"
    now = time.time()

    timestamps = _request_timestamps.get(limiter_key, [])
    # Evict timestamps older than the sliding window
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    
    if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for AI generation endpoints. Please retry after a brief delay.",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))}
        )

    timestamps.append(now)
    _request_timestamps[limiter_key] = timestamps
