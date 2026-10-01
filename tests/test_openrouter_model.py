"""Model-layer tests against a local mock HTTP server."""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from typing import Any

import pytest
import tenacity
from game_arena.harness import model_generation

import openrouter_model
from openrouter_model import OpenRouterModel


class _MockServer:
  """Threaded HTTP server that records requests and serves scripted replies."""

  def __init__(self):
    self.requests: list[dict[str, Any]] = []
    self.replies: list[tuple[int, dict[str, Any] | str]] = []
    server = self

    class Handler(http.server.BaseHTTPRequestHandler):
      def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        server.requests.append({
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": json.loads(body),
        })
        status, payload = server.replies.pop(0) if server.replies else (200, _ok())
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

      def log_message(self, *args):  # silence
        pass

    class Server(http.server.ThreadingHTTPServer):
      def server_bind(self):  # Skip the slow reverse-DNS lookup in HTTPServer.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    self.httpd = Server(("127.0.0.1", 0), Handler)
    self.thread = threading.Thread(
        target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    self.thread.start()

  @property
  def url(self) -> str:
    host, port = self.httpd.server_address
    return f"http://{host}:{port}/api/v1/chat/completions"

  def close(self):
    self.httpd.shutdown()
    self.httpd.server_close()


def _ok(content="Final Answer: FOLD", reasoning=None, provider="DeepInfra", **usage):
  message: dict[str, Any] = {"role": "assistant", "content": content}
  if reasoning is not None:
    message["reasoning"] = reasoning
  return {
      "id": "gen-1",
      "provider": provider,
      "model": "some/model",
      "choices": [{"message": message, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                "cost": 0.00042, "completion_tokens_details": {"reasoning_tokens": 12}}
      | usage,
  }


@pytest.fixture
def server():
  s = _MockServer()
  yield s
  s.close()


@pytest.fixture(autouse=True)
def fast_retry():
  """Disable the harness's exponential backoff for the wrapped generate methods."""
  saved = []
  for name in ("generate_with_text_input", "generate_with_image_text_input"):
    fn = getattr(OpenRouterModel, name)
    retry = fn.retry
    saved.append((retry, retry.wait, retry.stop))
    retry.wait = tenacity.wait_none()
    retry.stop = tenacity.stop_after_attempt(4)
  yield
  for retry, wait, stop in saved:
    retry.wait, retry.stop = wait, stop


def _model(server, **kw):
  return OpenRouterModel("openai/gpt-5-mini", api_key="test-key", base_url=server.url, **kw)


def test_request_shape_and_headers(server):
  model = _model(server, model_options={"temperature": 0.3, "top_p": 0.9, "top_k": 40,
                                        "max_tokens": 2048})
  out = model.generate_with_text_input(model_generation.ModelTextInput(
      prompt_text="Hello", system_instruction="You are seat 0."))
  assert len(server.requests) == 1
  req = server.requests[0]
  assert req["path"] == "/api/v1/chat/completions"
  assert req["headers"]["Authorization"] == "Bearer test-key"
  assert req["headers"]["Content-Type"] == "application/json"
  body = req["body"]
  assert body["model"] == "openai/gpt-5-mini"
  assert body["messages"] == [
      {"role": "system", "content": "You are seat 0."},
      {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
  ]
  assert body["max_tokens"] == 2048
  assert body["temperature"] == 0.3 and body["top_p"] == 0.9 and body["top_k"] == 40
  assert body["usage"] == {"include": True}
  assert "provider" not in body
  assert out.main_response == "Final Answer: FOLD"
  assert out.request_for_logging == body
  assert out.duration_success_only_secs is not None


def test_default_max_tokens_is_high(server):
  _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert server.requests[0]["body"]["max_tokens"] == openrouter_model.DEFAULT_MAX_TOKENS
  assert openrouter_model.DEFAULT_MAX_TOKENS >= 8192


def test_provider_pinning_passthrough_and_served_provider(server):
  provider = {"order": ["DeepInfra"], "allow_fallbacks": False}
  model = _model(server, model_options={"provider": provider, "reasoning": {"effort": "low"}})
  server.replies.append((200, _ok(provider="DeepInfra")))
  out = model.generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  body = server.requests[0]["body"]
  assert body["provider"] == provider
  assert body["reasoning"] == {"effort": "low"}
  assert openrouter_model.served_provider(out) == "DeepInfra"


def test_retry_on_5xx_then_success(server):
  server.replies.extend([(500, {"error": "boom"}), (503, "not json"), (200, _ok())])
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert len(server.requests) == 3
  assert out.main_response == "Final Answer: FOLD"


def test_retry_on_429(server):
  server.replies.extend([(429, {"error": {"message": "rate limited", "code": 429}}), (200, _ok())])
  _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert len(server.requests) == 2


def test_no_retry_on_4xx(server):
  server.replies.append((400, {"error": {"message": "bad request", "code": 400}}))
  with pytest.raises(model_generation.DoNotRetryError) as excinfo:
    _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert len(server.requests) == 1
  assert "bad request" in str(excinfo.value)


def test_no_retry_on_401(server):
  server.replies.append((401, {"error": {"message": "no key", "code": 401}}))
  with pytest.raises(model_generation.DoNotRetryError):
    _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert len(server.requests) == 1


def test_error_body_with_200_is_retried(server):
  server.replies.extend([(200, {"error": {"message": "provider overloaded", "code": 502}}),
                         (200, _ok())])
  _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert len(server.requests) == 2


def test_token_usage_and_cost_capture(server):
  server.replies.append((200, _ok(prompt_tokens=1000, completion_tokens=250, total_tokens=1250)))
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert out.prompt_tokens == 1000
  assert out.generation_tokens == 250
  assert out.total_tokens == 1250
  assert out.reasoning_tokens == 12
  assert openrouter_model.served_cost(out) == pytest.approx(0.00042)


def test_reasoning_capture(server):
  server.replies.append((200, _ok(content="Final Answer: TRUCO", reasoning="they look weak")))
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert out.main_response == "Final Answer: TRUCO"
  assert out.main_response_and_thoughts == "<think>they look weak</think>Final Answer: TRUCO"


def test_reasoning_details_fallback_and_think_tags(server):
  reply = _ok(content="<think>hmm</think>Final Answer: FOLD")
  reply["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.text", "text": "d1"}]
  server.replies.append((200, reply))
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert out.main_response_and_thoughts.startswith("<think>d1</think>")
  # Inline think tags are stripped from main_response when no API reasoning.
  server.replies.append((200, _ok(content="<think>hmm</think>Final Answer: FOLD")))
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert out.main_response == "Final Answer: FOLD"


def test_none_content_becomes_empty_string(server):
  server.replies.append((200, _ok(content=None)))
  out = _model(server).generate_with_text_input(model_generation.ModelTextInput(prompt_text="x"))
  assert out.main_response == ""


def test_missing_api_key_raises(monkeypatch):
  monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
  with pytest.raises(ValueError):
    OpenRouterModel("openai/gpt-5-mini")
  # Self-hosted endpoints do not need a key.
  OpenRouterModel("local", base_url="http://localhost:8000/v1/chat/completions")


def test_image_text_input(server):
  model = _model(server)
  out = model.generate_with_image_text_input(model_generation.ModelImageTextInput(
      prompt_text="look", prompt_image_bytes=b"abc", prompt_image_mime_type="image/png"))
  content = server.requests[0]["body"]["messages"][0]["content"]
  assert content[0] == {"type": "text", "text": "look"}
  assert content[1]["type"] == "image_url"
  assert out.main_response == "Final Answer: FOLD"
