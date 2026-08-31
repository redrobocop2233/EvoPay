"""Reliable Gemini GenAI client with Pydantic-validated structured output."""
import json
import os
import time
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, create_model

load_dotenv(override=True)

T = TypeVar("T", bound=BaseModel)

# 3.7 is the primary model; fallbacks protect the prototype from temporary
# model-specific capacity errors. All are current stable/available model IDs.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]


class GenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = 1,
        timeout_ms: int = 90000,
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No GEMINI_API_KEY found. Add it to .env or set it in the environment."
            )

        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self.model = model
        self.max_retries = max_retries

    def complete_json_list(
        self,
        system: str,
        user: str,
        item_schema: Type[T],
        max_tokens: int = 4000,
    ) -> list[T]:
        Envelope = create_model(
            "StructuredItems",
            items=(list[item_schema], ...),
        )

        raw = self._call(system, user, Envelope, max_tokens)
        data = self._extract_json(raw)
        items = data.get("items") if isinstance(data, dict) else data

        if not isinstance(items, list):
            raise ValueError(
                f"Expected a JSON array/items field, got {type(items).__name__}"
            )

        validated = []
        errors = []

        for i, item in enumerate(items):
            try:
                validated.append(item_schema.model_validate(item))
            except ValidationError as exc:
                errors.append(f"item {i}: {exc}")

        if not validated and errors:
            raise ValueError(f"No items validated. Errors: {errors}")

        return validated

    def complete_json_single(
        self,
        system: str,
        user: str,
        schema: Type[T],
        max_tokens: int = 1500,
    ) -> T:
        raw = self._call(system, user, schema, max_tokens)
        data = self._extract_json(raw)

        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            retry_user = (
                user
                + "\n\nPrevious output failed validation:\n"
                + str(exc)
                + "\nReturn corrected JSON only matching the schema exactly."
            )
            raw2 = self._call(system, retry_user, schema, max_tokens)
            return schema.model_validate(self._extract_json(raw2))

    def _call(
        self,
        system: str,
        user: str,
        schema: Type[BaseModel],
        max_tokens: int,
    ) -> str:
        from google.genai import types

        prompt = system + "\n\nUSER TASK:\n" + user

        models_to_try = []
        for model in [self.model, *FALLBACK_MODELS]:
            if model not in models_to_try:
                models_to_try.append(model)

        last_error = None

        for model in models_to_try:
            for attempt in range(self.max_retries + 1):
                try:
                    print(f"   Gemini call: {model} (attempt {attempt + 1}/{self.max_retries + 1})", flush=True)
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=schema,
                            max_output_tokens=max_tokens,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                disable=True
                            ),
                        ),
                    )

                    if not response.text:
                        raise RuntimeError(
                            f"Gemini returned an empty response from {model}."
                        )

                    # Remember the model that actually worked.
                    self.model = model
                    return response.text

                except Exception as exc:
                    last_error = exc

                    # Only retry transient service/rate/capacity failures.
                    status = getattr(exc, "status_code", None)
                    message = str(exc).lower()

                    # The Gemini SDK may surface transport failures as httpx
                    # exceptions instead of an HTTP status. Treat those as
                    # transient too; otherwise a server-side disconnect can
                    # bypass our model fallback logic.
                    exc_name = type(exc).__name__.lower()
                    exc_module = type(exc).__module__.lower()
                    transport_failure = (
                        "remoteprotocolerror" in exc_name
                        or "connecterror" in exc_name
                        or "readtimeout" in exc_name
                        or "writeerror" in exc_name
                        or "pooltimeout" in exc_name
                        or "timeout" in exc_name
                        or exc_module.startswith("httpx")
                    )

                    transient = (
                        transport_failure
                        or status in {408, 429, 500, 502, 503, 504}
                        or "408" in message
                        or "429" in message
                        or "500" in message
                        or "502" in message
                        or "503" in message
                        or "504" in message
                        or "deadline_exceeded" in message
                        or "deadline expired" in message
                        or "unavailable" in message
                        or "high demand" in message
                        or "temporarily" in message
                        or "rate limit" in message
                        or "server disconnected" in message
                    )

                    if not transient:
                        raise

                    if attempt < self.max_retries:
                        # 1s, 2s, 4s...
                        time.sleep(2 ** attempt)

            # If this model is temporarily unavailable, move to fallback.

        raise RuntimeError(
            "All configured Gemini models were unavailable after retries. "
            f"Last error: {last_error}"
        ) from last_error

    @staticmethod
    def _extract_json(raw: str):
        text = (raw or "").strip()

        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text

            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]

        return json.loads(text.strip())
