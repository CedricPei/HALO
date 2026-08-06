"""Live web search through Tavily."""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_env():
    """Load the project .env without replacing existing variables."""
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class Web:
    """Run live Tavily searches without storing local state."""

    def __init__(self, api_key=None, timeout=30):
        _load_env()
        self.api_key = api_key
        self.timeout = timeout

    def search_web(self, query, max_results=5):
        """Return Tavily results with title, URL, page content, and score."""
        api_key = self.api_key or os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY is required"}

        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(
                {
                    "query": query,
                    "max_results": max_results,
                    "include_raw_content": "markdown",
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            return {"error": f"Tavily request failed: {error}"}

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("raw_content") or item.get("content", ""),
                "score": item.get("score", 0),
            }
            for item in data.get("results", [])
        ]

    def open_url(self, url):
        """Open a URL and return its final URL, status, and decoded content."""
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                return {
                    "url": response.geturl(),
                    "status": response.status,
                    "content": response.read().decode(encoding, errors="replace"),
                }
        except urllib.error.URLError as error:
            return {"error": f"URL request failed: {error}"}
