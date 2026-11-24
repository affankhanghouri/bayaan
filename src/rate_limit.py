"""Share the Gemini request budget across graph nodes and speech helpers."""
import asyncio
import os
import threading
import time
from collections import deque
from langchain_core.rate_limiters import BaseRateLimiter


class GeminiRateLimiter(BaseRateLimiter):
    def __init__(self):
        self.limit = max(1, int(os.getenv('GEMINI_REQUESTS_PER_MINUTE', '5')))
        self.window = 61.0
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self, *, blocking=True):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.calls and self.calls[0] <= now - self.window:
                    self.calls.popleft()
                if len(self.calls) < self.limit:
                    self.calls.append(now)
                    return True
                delay = self.window - (now - self.calls[0])
            if not blocking:
                return False
            time.sleep(max(.01, delay))

    async def aacquire(self, *, blocking=True):
        return await asyncio.to_thread(self.acquire, blocking=blocking)


gemini_limiter = GeminiRateLimiter()
