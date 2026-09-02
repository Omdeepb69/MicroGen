"""Integration unit tests for microgen FastAPI HTTP endpoints."""

import threading
import time
import pytest
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from microgen.backends import PyTorchBackend
from microgen.devices import get_device
from microgen.runtime import KVCacheManager
from microgen.scheduler import ContinuousBatchingScheduler
from microgen.api import create_app


@pytest.fixture(scope="module")
def api_client():
    device = get_device("cpu")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    backend = PyTorchBackend(device=device)
    backend.load_model("sshleifer/tiny-gpt2", model_instance=model)
    kv_cache_manager = KVCacheManager()

    scheduler = ContinuousBatchingScheduler(
        backend=backend,
        kv_cache_manager=kv_cache_manager,
        eos_token_id=tokenizer.eos_token_id or 50256,
        max_batch_size=4,
    )

    # Start background scheduler step thread
    running = True

    def scheduler_loop():
        while running:
            scheduler.step()
            time.sleep(0.005)

    thread = threading.Thread(target=scheduler_loop, daemon=True)
    thread.start()

    app = create_app(scheduler=scheduler, tokenizer=tokenizer)
    client = TestClient(app)

    yield client

    running = False


def test_health_check(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models(api_client):
    response = api_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "microgen-model"


def test_chat_completions_non_streaming(api_client):
    payload = {
        "model": "microgen-model",
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 10,
        "stream": False,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "content" in data["choices"][0]["message"]
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completions_streaming(api_client):
    payload = {
        "model": "microgen-model",
        "messages": [{"role": "user", "content": "Stream test"}],
        "max_tokens": 8,
        "stream": True,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    lines = response.text.strip().split("\n\n")
    assert len(lines) > 0
    assert lines[-1] == "data: [DONE]"


def test_completions_non_streaming(api_client):
    payload = {
        "model": "microgen-model",
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "stream": False,
    }
    response = api_client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]
