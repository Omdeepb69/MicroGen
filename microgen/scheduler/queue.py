"""Request tracking dataclasses and thread-safe RequestQueue for LLM serving."""

from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from typing import Dict, List, Optional


class RequestStatus(str, Enum):
    """Lifecycle statuses for an inference request."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Request:
    """Dataclass representing a single generation request and tracking its state & metrics."""

    request_id: str
    prompt: str
    prompt_ids: List[int]
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    priority: int = 0  # Higher value indicates higher scheduling priority
    status: RequestStatus = RequestStatus.PENDING
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    generated_token_ids: List[int] = field(default_factory=list)
    generated_text: str = ""
    error_message: Optional[str] = None

    @property
    def is_finished(self) -> bool:
        """Return True if request execution has terminated."""
        return self.status in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        )

    @property
    def num_prompt_tokens(self) -> int:
        """Return count of tokens in prompt."""
        return len(self.prompt_ids)

    @property
    def num_generated_tokens(self) -> int:
        """Return count of generated tokens so far."""
        return len(self.generated_token_ids)

    @property
    def total_sequence_length(self) -> int:
        """Return total sequence length (prompt + generated)."""
        return self.num_prompt_tokens + self.num_generated_tokens


class RequestQueue:
    """Thread-safe queue managing incoming generation requests with priority support."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: Dict[str, Request] = {}
        self._pending_ids: List[str] = []
        self._dirty: bool = False

    def enqueue(self, request: Request) -> None:
        """Add request to queue."""
        with self._lock:
            self._requests[request.request_id] = request
            self._pending_ids.append(request.request_id)
            self._dirty = True

    def _ensure_sorted_unlocked(self) -> None:
        """Sort pending IDs lazily by priority (descending) and arrival_time (ascending)."""
        if self._dirty:
            self._pending_ids.sort(
                key=lambda req_id: (
                    -self._requests[req_id].priority,
                    self._requests[req_id].arrival_time,
                )
            )
            self._dirty = False

    def dequeue(self) -> Optional[Request]:
        """Fetch and remove the next highest-priority pending request."""
        with self._lock:
            self._ensure_sorted_unlocked()
            while self._pending_ids:
                req_id = self._pending_ids.pop(0)
                req = self._requests.get(req_id)
                if req and req.status == RequestStatus.PENDING:
                    req.status = RequestStatus.RUNNING
                    req.start_time = time.time()
                    return req
            return None

    def pop_batch(self, max_batch_size: int) -> List[Request]:
        """Fetch up to max_batch_size pending requests as a batch."""
        batch: List[Request] = []
        with self._lock:
            self._ensure_sorted_unlocked()
            while len(batch) < max_batch_size and self._pending_ids:
                req_id = self._pending_ids.pop(0)
                req = self._requests.get(req_id)
                if req and req.status == RequestStatus.PENDING:
                    req.status = RequestStatus.RUNNING
                    req.start_time = time.time()
                    batch.append(req)
        return batch

    def cancel(self, request_id: str) -> bool:
        """Cancel a request by ID."""
        with self._lock:
            req = self._requests.get(request_id)
            if req and not req.is_finished:
                req.status = RequestStatus.CANCELLED
                req.finish_time = time.time()
                if request_id in self._pending_ids:
                    self._pending_ids.remove(request_id)
                return True
            return False

    def get(self, request_id: str) -> Optional[Request]:
        """Lookup request by ID."""
        with self._lock:
            return self._requests.get(request_id)

    def get_pending_requests(self) -> List[Request]:
        """Return list of all currently PENDING requests."""
        with self._lock:
            self._ensure_sorted_unlocked()
            return [
                self._requests[req_id]
                for req_id in self._pending_ids
                if self._requests[req_id].status == RequestStatus.PENDING
            ]

    def get_running_requests(self) -> List[Request]:
        """Return list of all currently RUNNING requests."""
        with self._lock:
            return [
                req for req in self._requests.values() if req.status == RequestStatus.RUNNING
            ]

    def size(self) -> int:
        """Return count of pending requests."""
        with self._lock:
            return len(self._pending_ids)

    def is_empty(self) -> bool:
        """Return True if no pending requests exist."""
        with self._lock:
            return len(self._pending_ids) == 0
