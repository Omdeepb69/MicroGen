"""Thread-safe token bucket rate limiter for RPM (requests per minute) and TPM (tokens per minute)."""

import threading
import time
from typing import Optional


class TokenBucketRateLimiter:
    """Thread-safe rate limiter enforcing RPM (requests/min) and TPM (tokens/min) limits."""

    def __init__(
        self,
        max_rpm: Optional[float] = None,
        max_tpm: Optional[float] = None,
    ) -> None:
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm

        self._lock = threading.Lock()
        self._last_update_time = time.time()

        # Current available token balances
        self._request_tokens: float = float(max_rpm) if max_rpm is not None else float("inf")
        self._token_tokens: float = float(max_tpm) if max_tpm is not None else float("inf")

    def _refill(self) -> None:
        """Refill bucket balances based on elapsed time since last update (called under lock)."""
        now = time.time()
        elapsed = now - self._last_update_time
        self._last_update_time = now

        if self.max_rpm is not None:
            # Refill rate per second = max_rpm / 60.0
            refill_rpm = elapsed * (self.max_rpm / 60.0)
            self._request_tokens = min(self.max_rpm, self._request_tokens + refill_rpm)

        if self.max_tpm is not None:
            # Refill rate per second = max_tpm / 60.0
            refill_tpm = elapsed * (self.max_tpm / 60.0)
            self._token_tokens = min(self.max_tpm, self._token_tokens + refill_tpm)

    def check(self, num_requests: int = 1, num_tokens: int = 0) -> bool:
        """Non-consuming check if the specified request/token quota is available."""
        with self._lock:
            self._refill()
            req_ok = self._request_tokens >= num_requests
            tok_ok = self._token_tokens >= num_tokens
            return req_ok and tok_ok

    def acquire(self, num_requests: int = 1, num_tokens: int = 0) -> bool:
        """Acquire quota for requests and tokens. Consumes tokens if available and returns True, else False."""
        with self._lock:
            self._refill()
            req_ok = self._request_tokens >= num_requests
            tok_ok = self._token_tokens >= num_tokens

            if req_ok and tok_ok:
                if self.max_rpm is not None:
                    self._request_tokens -= num_requests
                if self.max_tpm is not None:
                    self._token_tokens -= num_tokens
                return True

            return False

    def reset(self) -> None:
        """Reset rate limiter token buckets to full capacity."""
        with self._lock:
            self._last_update_time = time.time()
            self._request_tokens = float(self.max_rpm) if self.max_rpm is not None else float("inf")
            self._token_tokens = float(self.max_tpm) if self.max_tpm is not None else float("inf")
