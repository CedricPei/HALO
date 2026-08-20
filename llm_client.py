"""Shared non-streaming HTTP(S) client for construction and evaluation."""

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"
LOGGER = logging.getLogger(__name__)
OPENAI_CONSTRUCTION_MODEL = "gpt-5.6-sol"
OPENAI_CONSTRUCTION_REASONING_EFFORT = "xhigh"


class LLMTransportError(RuntimeError):
    """An HTTP(S)-layer failure before a usable response."""


class LLMProtocolError(RuntimeError):
    """A successful HTTP response with an unusable model payload."""

    def __init__(self, message):
        super().__init__(message)
        self.call = None


def load_env():
    """Load simple KEY=VALUE entries without replacing existing variables."""
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _extract_openai_text(data):
    output_texts, final_texts = [], []
    for output in data.get("output", []):
        if output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if item.get("type") == "refusal":
                raise LLMProtocolError(f"LLM refused the request: {item['refusal']}")
            if item.get("type") == "output_text":
                output_texts.append(item["text"])
                if output.get("phase") == "final_answer":
                    final_texts.append(item["text"])
    if final_texts:
        return final_texts[-1]
    if output_texts:
        return output_texts[-1]
    raise LLMProtocolError("LLM response did not contain output text")


def _extract_gemini_text(data):
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        reason = feedback.get("blockReason") or "no candidates"
        raise LLMProtocolError(
            f"Gemini response did not contain a candidate: {reason}"
        )
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [part["text"] for part in parts if isinstance(part.get("text"), str)]
    if texts:
        return "".join(texts)
    finish_reason = candidates[0].get("finishReason") or "unknown"
    raise LLMProtocolError(
        "Gemini response did not contain output text "
        f"(finish reason: {finish_reason})"
    )


def _first_json_object(text):
    """Extract the first balanced JSON object, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _remove_trailing_commas(text):
    """Remove commas immediately before a closing object or array delimiter."""
    output = []
    in_string = escaped = False
    for index, character in enumerate(text):
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == ",":
            following = text[index + 1:].lstrip()
            if following.startswith(("}", "]")):
                continue
        output.append(character)
    return "".join(output)


def _parse_model_json(raw_text):
    """Parse an object after harmless wrapper and trailing-comma cleanup."""
    if not isinstance(raw_text, str):
        raise TypeError("model output is not text")
    text = raw_text.strip().lstrip("\ufeff")
    extracted = _first_json_object(text)
    candidates = [text]
    if extracted and extracted != text:
        candidates.append(extracted)
    last_error = None
    for candidate in candidates:
        for normalized in (candidate, _remove_trailing_commas(candidate)):
            try:
                return json.loads(normalized)
            except json.JSONDecodeError as error:
                last_error = error
    raise last_error


class LLMClient:
    """Use one transport implementation for Creator and evaluated-Agent calls."""

    def __init__(self, temperature=None):
        load_env()
        self.provider = os.getenv("CREATOR_LLM_PROVIDER", "openai").lower()
        if self.provider not in {"openai", "gemini"}:
            raise ValueError("LLM provider must be 'openai' or 'gemini'")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_responses_url = os.getenv("OPENAI_RESPONSES_URL", "")
        self.openai_chat_completions_url = os.getenv(
            "OPENAI_CHAT_COMPLETIONS_URL", ""
        )
        self.gemini_base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        ).rstrip("/")
        self.model = (
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            if self.provider == "gemini"
            else OPENAI_CONSTRUCTION_MODEL
        )
        self.temperature = temperature
        self.timeout = float(os.getenv("LLM_TIMEOUT", "300"))
        self.construction_reasoning_effort = os.getenv(
            "OPENAI_REASONING_EFFORT",
            OPENAI_CONSTRUCTION_REASONING_EFFORT,
        )
        self.thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "high")
        self.gemini_transport = os.getenv(
            "EVALUATION_GEMINI_TRANSPORT", "developer"
        )
        configured_max_tokens = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")
        self.max_output_tokens = (
            int(configured_max_tokens) if configured_max_tokens else None
        )

        if self.gemini_transport not in {"developer", "openai-chat"}:
            raise ValueError("Gemini transport must be 'developer' or 'openai-chat'")
        if self.construction_reasoning_effort not in {
            "none", "low", "medium", "high", "xhigh", "max"
        }:
            raise ValueError("Invalid OpenAI construction reasoning effort")
        if self.thinking_level not in {"minimal", "low", "medium", "high"}:
            raise ValueError("Invalid Gemini thinking level")
        for url in filter(
            None,
            (
                self.openai_responses_url,
                self.openai_chat_completions_url,
                self.gemini_base_url,
            ),
        ):
            if urllib.parse.urlsplit(url).scheme.lower() not in {"http", "https"}:
                raise ValueError("LLM URLs must use HTTP or HTTPS")

    def generate_text(self, prompt, request_label):
        """Return raw model text for a construction role."""
        if not self.model:
            raise ValueError(f"{self.provider.upper()} model is required")
        if self.provider == "gemini":
            configured = os.getenv("GEMINI_TEMPERATURE")
            temperature = (
                float(configured)
                if configured is not None
                else (1.0 if self.model.startswith("gemini-3") else self.temperature)
            )
            operation = lambda: self._gemini(self.model, prompt, temperature)
        else:
            operation = lambda: self._openai_responses(
                self.model,
                prompt,
                self.temperature,
                reasoning_effort=self.construction_reasoning_effort,
            )
        call = self._retry(operation, request_label)
        return call["raw_text"]

    def generate_json(self, prompt, model):
        """Return normalized JSON plus usage and latency for an evaluated Agent."""
        if model.startswith("gemini") and self.gemini_transport == "developer":
            operation = lambda: self._gemini(model, prompt, 1.0)
        elif model.startswith("gemini"):
            operation = lambda: self._openai_chat(model, prompt, 1.0)
        else:
            operation = lambda: self._openai_responses(model, prompt, None)
        call = self._retry(operation, model)
        try:
            call["parsed"] = _parse_model_json(call["raw_text"])
        except (json.JSONDecodeError, TypeError) as error:
            detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
            protocol_error = LLMProtocolError(
                f"model returned invalid JSON: {detail}"
            )
            protocol_error.call = call
            raise protocol_error from error
        return call

    def _retry(self, operation, label):
        for attempt in range(1, 6):
            try:
                return operation()
            except LLMTransportError as error:
                LOGGER.warning(
                    "[%s] HTTP attempt %s/5 failed; %s: %s",
                    label,
                    attempt,
                    "stopping" if attempt == 5 else "retrying immediately",
                    error,
                )
                if attempt == 5:
                    raise LLMTransportError(
                        "LLM call stopped after 5 network failures"
                    ) from error

    def _post(self, url, payload, headers):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise LLMTransportError(f"HTTP {error.code}: {detail}") from error
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise LLMTransportError(str(error)) from error
        return data, round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _openai_usage(data):
        usage = data.get("usage") or {}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get(
            "output_tokens", usage.get("completion_tokens", 0)
        ) or 0
        output_details = usage.get(
            "output_tokens_details", usage.get("completion_tokens_details")
        ) or {}
        reasoning_tokens = output_details.get("reasoning_tokens", 0) or 0
        return {
            "input_tokens": input_tokens,
            "completion_tokens": max(output_tokens - reasoning_tokens, 0),
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": usage.get(
                "total_tokens", input_tokens + output_tokens
            ) or 0,
        }

    @staticmethod
    def _gemini_usage(data):
        usage = data.get("usageMetadata") or {}
        input_tokens = usage.get("promptTokenCount", 0) or 0
        completion_tokens = usage.get("candidatesTokenCount", 0) or 0
        reasoning_tokens = usage.get("thoughtsTokenCount", 0) or 0
        return {
            "input_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": usage.get(
                "totalTokenCount",
                input_tokens + completion_tokens + reasoning_tokens,
            ) or 0,
        }

    @staticmethod
    def _call(text, usage, latency, endpoint, response_id):
        return {
            "raw_text": text,
            "usage": usage,
            "latency_ms": latency,
            "endpoint": endpoint,
            "response_id": response_id,
        }

    def _openai_responses(
        self, model, prompt, temperature, reasoning_effort=None
    ):
        if not self.openai_api_key or not self.openai_responses_url:
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_RESPONSES_URL are required"
            )
        payload = {
            "model": model,
            "input": prompt,
            "store": False,
            "text": {"format": {"type": "json_object"}},
        }
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            payload["temperature"] = temperature
        data, latency = self._post(
            self.openai_responses_url,
            payload,
            {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        usage = self._openai_usage(data)
        try:
            text = _extract_openai_text(data)
        except LLMProtocolError as error:
            error.call = self._call(
                None, usage, latency, "openai/responses", data.get("id")
            )
            raise
        return self._call(
            text, usage, latency, "openai/responses", data.get("id")
        )

    def _openai_chat(self, model, prompt, temperature):
        if not self.openai_api_key or not self.openai_chat_completions_url:
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_CHAT_COMPLETIONS_URL are required"
            )
        data, latency = self._post(
            self.openai_chat_completions_url,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
            {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        usage = self._openai_usage(data)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            protocol_error = LLMProtocolError(
                "Chat Completions response did not contain output text"
            )
            protocol_error.call = self._call(
                None, usage, latency, "openai/chat_completions", data.get("id")
            )
            raise protocol_error from error
        return self._call(
            text,
            usage,
            latency,
            "openai/chat_completions",
            data.get("id"),
        )

    def _gemini(self, model, prompt, temperature):
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        model = model.removeprefix("models/")
        config = {"responseMimeType": "application/json"}
        if temperature is not None and not model.startswith("gemini-3"):
            config["temperature"] = temperature
        if self.max_output_tokens is not None:
            config["maxOutputTokens"] = self.max_output_tokens
        if model.startswith("gemini-3"):
            config["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        else:
            budgets = {
                "minimal": 1024,
                "low": 1024,
                "medium": 8192,
                "high": 24576,
            }
            config["thinkingConfig"] = {
                "thinkingBudget": budgets[self.thinking_level]
            }
        model_path = urllib.parse.quote(model, safe="-._")
        data, latency = self._post(
            f"{self.gemini_base_url}/models/{model_path}:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": config,
            },
            {
                "x-goog-api-key": self.gemini_api_key,
                "Content-Type": "application/json",
            },
        )
        usage = self._gemini_usage(data)
        try:
            text = _extract_gemini_text(data)
        except LLMProtocolError as error:
            error.call = self._call(
                None,
                usage,
                latency,
                "gemini/generateContent",
                data.get("responseId"),
            )
            raise
        return self._call(
            text,
            usage,
            latency,
            "gemini/generateContent",
            data.get("responseId"),
        )
