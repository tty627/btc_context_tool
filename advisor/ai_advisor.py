"""Thin wrapper around the OpenAI chat-completions API for market analysis."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from ..config import (
        OPENAI_FALLBACK_MODELS,
        OPENAI_MAX_TOKENS,
        OPENAI_MODEL,
        OPENAI_TEMPERATURE,
    )
    from ..reports.prompt_generator import PromptGenerator
except ImportError:
    from config import OPENAI_FALLBACK_MODELS, OPENAI_MAX_TOKENS, OPENAI_MODEL, OPENAI_TEMPERATURE
    from reports.prompt_generator import PromptGenerator

logger = logging.getLogger("btc_context.ai")

_OPENAI_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
_VALID_REASONING_EFFORT = frozenset({"none", "low", "medium", "high", "xhigh"})
_RETRY_REASONING_EFFORT = {
    "xhigh": "medium",
    "high": "low",
    "medium": "low",
    "low": "none",
}
_MODEL_NOT_FOUND_MARKERS = (
    "model_not_found",
    "does not exist or you do not have access",
)
_BODY_PARSE_ERROR_MARKERS = (
    "could not parse the json body of your request",
    "expects a json payload",
)


def _normalize_reasoning_effort(model: str, override: Optional[str] = None) -> Optional[str]:
    m = (model or "").lower()
    if "gpt-5" not in m:
        return None
    raw = override if override is not None else os.getenv(_OPENAI_REASONING_EFFORT_ENV, "high")
    effort = raw.strip().lower()
    if effort not in _VALID_REASONING_EFFORT:
        effort = "high"
    return effort


def _openai_reasoning_kwargs(model: str, override: Optional[str] = None) -> dict:
    """GPT-5 系列：用 reasoning_effort 开启 Thinking（官方文档中的 Reasoning 档位）。"""
    effort = _normalize_reasoning_effort(model, override=override)
    if not effort:
        return {}
    logger.info("OpenAI reasoning_effort=%s (model=%s)", effort, model)
    return {"reasoning_effort": effort}


def _retry_reasoning_effort(model: str, current_effort: Optional[str]) -> Optional[str]:
    m = (model or "").lower()
    if "gpt-5" not in m or not current_effort:
        return None
    return _RETRY_REASONING_EFFORT.get(current_effort)


def _stringify_message_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _reasoning_tokens_exhausted(
    finish_reason: Optional[str],
    completion_tokens: Optional[int],
    reasoning_tokens: Optional[int],
) -> bool:
    if finish_reason != "length":
        return False
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return False
    if not isinstance(reasoning_tokens, int) or reasoning_tokens <= 0:
        return False
    return reasoning_tokens >= completion_tokens


def _empty_response_error(
    model: str,
    finish_reason: Optional[str],
    completion_tokens: Optional[int],
    reasoning_tokens: Optional[int],
) -> str:
    return (
        "OpenAI API returned empty response "
        f"(model={model}, finish_reason={finish_reason}, "
        f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens})"
    )


def _is_model_not_found_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    if any(marker in lowered for marker in _MODEL_NOT_FOUND_MARKERS):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code != 404:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    code = str(err.get("code", "")).lower()
    msg = str(err.get("message", "")).lower()
    return code == "model_not_found" or any(marker in msg for marker in _MODEL_NOT_FOUND_MARKERS)


def _is_body_parse_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return all(marker in lowered for marker in _BODY_PARSE_ERROR_MARKERS)


# 单张图过大则跳过（避免请求体爆炸）
_MAX_CHART_BYTES = 12 * 1024 * 1024

VISION_INTRO = (
    "【附图说明】下列 PNG 为当前快照的多周期 K 线结构图（及现货/永续对比图，若有）。"
    "请结合图形结构与下文文本数据面板（SECTION 1/2）综合判断趋势、关键位与可执行方案。\n\n"
)


def _png_data_uri(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_user_content(
    prompt: str,
    chart_items: Sequence[Tuple[str, str]],
) -> Union[str, List[dict]]:
    """chart_items: (timeframe_label, filesystem_path)."""
    valid: List[Tuple[str, Path]] = []
    for label, p in chart_items:
        pp = Path(p)
        if not pp.is_file() or pp.suffix.lower() != ".png":
            logger.warning("skip missing chart: %s", p)
            continue
        if pp.stat().st_size > _MAX_CHART_BYTES:
            logger.warning("skip oversized chart %s (%d bytes)", pp.name, pp.stat().st_size)
            continue
        valid.append((label, pp))

    if not valid:
        return prompt

    parts: List[dict] = [
        {"type": "text", "text": VISION_INTRO + prompt},
    ]
    for label, pp in valid:
        parts.append({"type": "text", "text": f"── K 线图: {label} ──"})
        parts.append({"type": "image_url", "image_url": {"url": _png_data_uri(pp)}})
    return parts


class AIAdvisor:
    """Call OpenAI-compatible chat-completions API for market analysis.

    Supports OpenAI, DeepSeek, and any OpenAI-compatible endpoint via base_url.
    When *chart_items* are given, sends multimodal user message (vision).
    """

    DEFAULT_API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model: str = OPENAI_MODEL,
        fallback_models: Optional[Sequence[str]] = None,
        temperature: float = OPENAI_TEMPERATURE,
        max_tokens: int = OPENAI_MAX_TOKENS,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.fallback_models = tuple(
            m.strip()
            for m in (OPENAI_FALLBACK_MODELS if fallback_models is None else fallback_models)
            if isinstance(m, str) and m.strip()
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    def _candidate_models(self) -> List[str]:
        # DeepSeek uses its own model names; do not inject OpenAI fallbacks there.
        if self.base_url and "deepseek" in self.base_url.lower():
            return [self.model]

        ordered = [self.model, *self.fallback_models]
        seen = set()
        candidates: List[str] = []
        for item in ordered:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(item)
        return candidates

    def analyze(
        self,
        prompt: str,
        chart_items: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """Backward-compatible single-stage decision call."""
        system_prompt = PromptGenerator().build_system_prompt(
            stage="decision",
            include_vision=bool(chart_items),
        )
        return self.analyze_with_messages(
            system_prompt=system_prompt,
            user_prompt=prompt,
            chart_items=chart_items,
        )

    def build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        chart_items: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[dict]:
        """Build the exact message payload that will be sent to the model."""
        user_content: Union[str, List[dict]] = user_prompt
        if chart_items:
            user_content = _build_user_content(user_prompt, chart_items)
            if isinstance(user_content, list) and len(user_content) > 1:
                if self.base_url and "deepseek" in (self.base_url or "").lower():
                    logger.warning(
                        "当前为 DeepSeek 接口；若返回错误，请改用 "
                        "--ai-provider openai --ai-model gpt-4o-mini（或 gpt-4o）以支持看图。"
                    )
                logger.info(
                    "多模态请求: 已附加 %d 张 K 线图",
                    (len(user_content) - 1) // 2,
                )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def analyze_with_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        chart_items: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        return self.analyze_messages(
            self.build_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                chart_items=chart_items,
            )
        )

    def analyze_messages(self, messages: Sequence[dict]) -> str:
        original_model = self.model
        candidates = self._candidate_models()
        for idx, candidate in enumerate(candidates):
            self.model = candidate
            try:
                result = self._try_openai_library(messages)
                if result is not None:
                    return result
                return self._call_via_urllib(messages)
            except Exception as exc:
                if _is_body_parse_error(exc):
                    logger.warning(
                        "检测到 OpenAI SDK 请求体偶发损坏（model=%s），改用 urllib 重试一次同一请求",
                        candidate,
                    )
                    try:
                        return self._call_via_urllib(messages)
                    except Exception as fallback_exc:
                        exc = fallback_exc
                has_next = idx + 1 < len(candidates)
                if has_next and _is_model_not_found_error(exc):
                    logger.warning(
                        "模型 %s 不可用（model_not_found），自动回退到 %s",
                        candidate,
                        candidates[idx + 1],
                    )
                    continue
                self.model = original_model
                raise

        self.model = original_model
        raise RuntimeError("no available model candidates for AI analysis")

    @staticmethod
    def _sdk_response_meta(response) -> Tuple[str, Optional[str], Optional[str], Optional[int], Optional[int]]:
        msg = response.choices[0].message
        text = _stringify_message_content(getattr(msg, "content", None))
        refusal = getattr(msg, "refusal", None)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
        return text, refusal, finish_reason, reasoning_tokens, completion_tokens

    def _try_openai_library(self, messages: Sequence[dict]) -> Optional[str]:
        try:
            import openai  # noqa: F811
        except ImportError:
            return None

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**kwargs)

        def _send_request(reasoning_effort: Optional[str] = None):
            rk = _openai_reasoning_kwargs(self.model, override=reasoning_effort)
            req: dict = {
                "model": self.model,
                "messages": list(messages),
                "timeout": 300.0,
            }
            if rk:
                req["max_completion_tokens"] = max(self.max_tokens, 4096)
                req.update(rk)
            else:
                req["temperature"] = self.temperature
                req["max_tokens"] = self.max_tokens
            return client.chat.completions.create(**req)

        current_effort = _normalize_reasoning_effort(self.model)
        response = _send_request(current_effort)
        text, refusal, finish_reason, reasoning_tokens, completion_tokens = self._sdk_response_meta(response)
        if text:
            return text
        if refusal:
            return str(refusal)

        retry_effort = None
        if _reasoning_tokens_exhausted(finish_reason, completion_tokens, reasoning_tokens):
            retry_effort = _retry_reasoning_effort(self.model, current_effort)

        if retry_effort:
            logger.warning(
                "OpenAI returned empty text after consuming all completion tokens as reasoning "
                "(model=%s, finish_reason=%s, completion_tokens=%s, reasoning_tokens=%s); "
                "retrying once with reasoning_effort=%s",
                self.model,
                finish_reason,
                completion_tokens,
                reasoning_tokens,
                retry_effort,
            )
            response = _send_request(retry_effort)
            text, refusal, finish_reason, reasoning_tokens, completion_tokens = self._sdk_response_meta(response)
            if text:
                return text
            if refusal:
                return str(refusal)

        raise RuntimeError(
            _empty_response_error(
                self.model,
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        )

    def _call_via_urllib(self, messages: Sequence[dict]) -> str:
        rk = _openai_reasoning_kwargs(self.model)
        payload: dict = {
            "model": self.model,
            "messages": list(messages),
        }
        if rk:
            payload["max_completion_tokens"] = max(self.max_tokens, 4096)
            payload.update(rk)
        else:
            payload["temperature"] = self.temperature
            payload["max_tokens"] = self.max_tokens
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        api_url = (self.base_url.rstrip("/") + "/chat/completions") if self.base_url else self.DEFAULT_API_URL
        request = Request(
            api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        max_attempts = 2
        delay = 2.0
        import time

        for attempt in range(1, max_attempts + 1):
            try:
                with urlopen(request, timeout=300) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    choice = payload["choices"][0]
                    msg = choice["message"]
                    text = _stringify_message_content(msg.get("content"))
                    if text:
                        return text
                    if msg.get("refusal"):
                        return str(msg["refusal"])
                    usage = payload.get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    raise RuntimeError(
                        _empty_response_error(
                            self.model,
                            finish_reason=choice.get("finish_reason"),
                            completion_tokens=usage.get("completion_tokens"),
                            reasoning_tokens=details.get("reasoning_tokens"),
                        )
                    )
            except HTTPError as exc:
                try:
                    err_body = exc.read().decode("utf-8")
                except Exception:
                    err_body = str(exc.reason)
                raise RuntimeError(
                    f"OpenAI API HTTP {exc.code}: {err_body}"
                ) from exc
            except URLError as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"OpenAI API connection error: {exc.reason}"
                    ) from exc
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def estimate_tokens(text: str, num_images: int = 0) -> int:
        """Rough token estimate for text; images ~2k tokens each (high detail)."""
        cn_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        en_chars = len(text) - cn_chars
        base = int(cn_chars / 3.5 + en_chars / 4)
        return base + num_images * 2000
