"""Small OpenAI Responses API client used by the creators."""

import json
import http.client
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


class LLMTransportError(RuntimeError):
    """A temporary connection failure before a usable LLM response."""


def extract_output_text(data):
    """Return the final-answer text, not an intermediate commentary message."""
    output_texts = []
    final_texts = []
    for output in data.get("output", []):
        if output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if item.get("type") == "refusal":
                raise RuntimeError(f"LLM refused the request: {item['refusal']}")
            if item.get("type") == "output_text":
                output_texts.append(item["text"])
                if output.get("phase") == "final_answer":
                    final_texts.append(item["text"])
    if final_texts:
        return final_texts[-1]
    if output_texts:
        return output_texts[-1]
    raise RuntimeError("LLM response did not contain output text")


def load_env(path=ENV_PATH):
    """Load simple KEY=VALUE entries without replacing existing variables."""
    if not Path(path).exists():
        return
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class LLMClient:
    """Make one non-streaming model request and return its output text."""

    def __init__(
        self,
        api_key=None,
        model=None,
        base_url=None,
        temperature=0.7,
        timeout=300,
    ):
        load_env()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not self.model:
            raise ValueError("OPENAI_MODEL is required")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required")
        scheme = urllib.parse.urlsplit(self.base_url).scheme.lower()
        if scheme not in {"http", "https"}:
            raise ValueError(
                "OPENAI_BASE_URL must use HTTP or HTTPS; socket transports are not supported"
            )

    def generate_json(self, prompt):
        """Call the model once and request a JSON object as output."""
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/responses",
            data=json.dumps(
                {
                    "model": self.model,
                    "input": prompt,
                    "store": False,
                    "temperature": self.temperature,
                    "text": {"format": {"type": "json_object"}},
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"LLM request failed ({error.code}): {detail}"
            if error.code == 429 or error.code >= 500:
                raise LLMTransportError(message) from error
            raise RuntimeError(message) from error
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError) as error:
            raise LLMTransportError(f"LLM request failed: {error}") from error
        except json.JSONDecodeError as error:
            raise LLMTransportError("LLM response envelope was invalid JSON") from error

        return extract_output_text(data)
