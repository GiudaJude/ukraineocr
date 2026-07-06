from __future__ import annotations

import base64
import collections
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import requests
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Configuration ----------
load_dotenv()
OPENAI_MODEL_OCR = os.getenv("OPENAI_MODEL_OCR", "gpt-5.4-mini")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "low")
GEMINI_MODEL_OCR = os.getenv("GEMINI_MODEL_OCR", "gemini-2.5-pro")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OCR_FEW_SHOT_LIMIT = int(os.getenv("OCR_FEW_SHOT_LIMIT", "8"))
OCR_MAX_OUTPUT_TOKENS = int(os.getenv("OCR_MAX_OUTPUT_TOKENS", "32768"))
OCR_OUTPUT_ROOT = os.getenv("OCR_OUTPUT_ROOT")

SYSTEM_INSTRUCTION = (
    "ROLE: You are an expert paleographer specializing in 16th and 17th century "
    "Lviv council records and Latin/Old Polish legal scripts. "
    "TASK: Transcribe the provided image into a full-text format. "
    "TRANSCRIPTION RULES: "
    "1. Expand abbreviations. "
    "You will encounter many scribal abbreviations (such as marks above words like "
    "'Leopolien'). Do not transcribe these literally. Expand them into the "
    "grammatically correct Latin form (e.g. Leopoliensis, Leopolitanus) based on the "
    "surrounding sentence structure. "
    "2. Grounding. "
    "Strictly follow the expansion logic and formatting style demonstrated in the few-shot "
    "exemplars. "
    "3. Contextual accuracy. "
    "In the case of smears and smudges, use the context of the surrounding words and "
    "letters. "
    "4. Output only. "
    "Provide only the final transcribed text. Do not include any thoughts in your output "
    "5. No empty responses. "
    "You must output a transcription. Even if the page is degraded, provide your "
    "best reading in plain text. Do not mark uncertainty with brackets, question "
    "marks, or editorial notation."
)

OCR_PROMPT = (
    f"{SYSTEM_INSTRUCTION}\n\n"
    "Carefully transcribe all visible, forward-facing text. "
    "Do not translate or paraphrase. Preserve original line breaks and obvious "
    "hyphenations.\n\n"
    "As you transcribe, apply these tags:\n"
    "- Prefix each Latin segment with [LA]\n"
    "- Prefix each Old Polish segment with [PL]\n"
    "Output only the transcription."
)

GEMINI_SAFETY_SETTINGS: list[Any] = []
if genai_types is not None:
    GEMINI_SAFETY_SETTINGS = [
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

OPENAI_CLIENT: Any | None = None
GEMINI_CLIENT: Any | None = None

FEW_SHOT_DATA = [
    {"image_path": "exemplars/Reference1.JPG", "text_path": "exemplars/Reference1.txt"},
    {"image_path": "exemplars/Reference2.JPG", "text_path": "exemplars/Reference2.txt"},
    {"image_path": "exemplars/Reference3.JPG", "text_path": "exemplars/Reference3.txt"},
    {"image_path": "exemplars/Reference4.JPG", "text_path": "exemplars/Reference4.txt"},
    {"image_path": "exemplars/Abbreviation1.JPG", "text_path": "exemplars/Abbreviation1.txt"},
    {"image_path": "exemplars/Abbreviation2.JPG", "text_path": "exemplars/Abbreviation2.txt"},
    {"image_path": "exemplars/Abbreviation3.JPG", "text_path": "exemplars/Abbreviation3.txt"},
    {"image_path": "exemplars/Abbreviation4.JPG", "text_path": "exemplars/Abbreviation4.txt"},
]

OPENAI_FEW_SHOTS: list[dict[str, str]] = []
GEMINI_UPLOADED_FEW_SHOTS: list[dict[str, Any]] = []
RETRIABLE_STATUS = {429, 500, 502, 503, 504}


def get_provider_name() -> str:
    """Choose the first fully usable provider, preferring OpenAI."""
    if OPENAI_API_KEY and OpenAI is not None:
        return "openai"
    if GEMINI_API_KEY and genai is not None:
        return "gemini"
    if OPENAI_API_KEY and GEMINI_API_KEY:
        raise RuntimeError(
            "API keys are configured, but neither provider SDK is installed. "
            "Install `openai` or `google-genai`."
        )
    if OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is set, but the `openai` package is not installed. "
            "Run `pip install openai`."
        )
    if GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is set, but the `google-genai` package is not installed. "
            "Run `pip install google-genai`."
        )
    raise RuntimeError(
        "No API key is configured. Set OPENAI_API_KEY or GEMINI_API_KEY. "
        "If both are present, OpenAI is used by default."
    )


def get_openai_client() -> Any:
    global OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError(
            "The `openai` package is required for OPENAI_API_KEY. Run `pip install openai`."
        )
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    return OPENAI_CLIENT


def get_gemini_client() -> Any:
    global GEMINI_CLIENT
    if genai is None:
        raise RuntimeError(
            "The `google-genai` package is required for GEMINI_API_KEY. "
            "Run `pip install google-genai`."
        )
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return GEMINI_CLIENT


def log_empty(img_path: str, reason: str, log_path: str = "empty_responses.txt") -> None:
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"{img_path} reason for emptiness: {reason}\n")


def strip_tags(text: str) -> str:
    text = re.sub(r"\[(LA|PL)\]\s*", "", text)
    text = re.sub(r"\[Latin Name:\s*(.*?)\]", r"\1", text)
    text = re.sub(r"\[Polish Name:\s*(.*?)\]", r"\1", text)
    text = text.replace("?", "")
    return text.strip()


def should_skip_input_file(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    if path.is_dir():
        return True
    if path.suffix.lower() in {".txt", ".json"}:
        return True
    return any(marker in path.name for marker in (".tresh.", ".top.", ".bottom."))


def get_output_directory(input_dir: Path) -> Path:
    if not OCR_OUTPUT_ROOT:
        return input_dir

    output_root = Path(OCR_OUTPUT_ROOT)
    try:
        relative_dir = input_dir.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative_dir = Path(input_dir.name)

    output_dir = output_root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def image_path_to_data_url(image_path: Path) -> str:
    suffix_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = suffix_to_mime.get(image_path.suffix.lower(), "application/octet-stream")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def get_few_shot_examples() -> list[tuple[Path, str]]:
    examples: list[tuple[Path, str]] = []
    for item in FEW_SHOT_DATA:
        image_path = Path(item["image_path"])
        text_path = Path(item["text_path"])
        if image_path.exists() and text_path.exists():
            examples.append((image_path, text_path.read_text(encoding="utf-8").strip()))
        else:
            print(f"WARNING: Missing {image_path} or {text_path}. Skipping")
    if OCR_FEW_SHOT_LIMIT <= 0:
        return []
    return examples[:OCR_FEW_SHOT_LIMIT]


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: int):
        self.max_calls = max_calls
        self.per = per_seconds
        self.history: collections.deque[float] = collections.deque()

    def wait(self) -> None:
        now = time.time()
        while self.history and now - self.history[0] >= self.per:
            self.history.popleft()
        if len(self.history) >= self.max_calls:
            sleep_for = self.per - (now - self.history[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.history.append(time.time())


limiter = RateLimiter(max_calls=2, per_seconds=60)


def _status_from_exc(exc: Exception) -> int | None:
    for attr in ("status", "code", "http_status", "http_code"):
        value = getattr(exc, attr, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    response = getattr(exc, "response", None) or getattr(exc, "http_response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            try:
                return int(status_code)
            except (TypeError, ValueError):
                pass

    exc_text = str(exc)
    for code in RETRIABLE_STATUS:
        if str(code) in exc_text:
            return code
    return None


def _parse_retry_after_seconds(err: Exception) -> float | None:
    response = getattr(err, "response", None) or getattr(err, "http_response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                return None
    return None


def _is_network_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
        ),
    )


def _call_with_limit(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Rate-limit + robust retries for API calls."""
    attempt = 0
    base_sleep = 2.0
    cap_sleep = 90.0
    sleep = base_sleep

    while True:
        limiter.wait()
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            status = _status_from_exc(exc)
            is_retriable = (
                _is_network_error(exc)
                or (status in RETRIABLE_STATUS)
                or ("rate" in str(exc).lower())
                or ("overloaded" in str(exc).lower())
                or ("temporarily unavailable" in str(exc).lower())
            )

            print(f"[retry {attempt}] {type(exc).__name__} {status or ''} ? {exc}")
            tb = traceback.format_exc(limit=1).strip()
            if tb:
                print(tb)

            if not is_retriable:
                raise

            retry_after = _parse_retry_after_seconds(exc)
            if retry_after is not None:
                time.sleep(max(1.0, retry_after))
            else:
                upper = min(cap_sleep, sleep * 3.0)
                sleep = random.uniform(base_sleep, max(base_sleep, upper))
                time.sleep(sleep)

            if attempt >= 8:
                raise


def init_openai_few_shots() -> None:
    if OPENAI_FEW_SHOTS:
        return
    for image_path, transcription in get_few_shot_examples():
        OPENAI_FEW_SHOTS.append(
            {"image_url": image_path_to_data_url(image_path), "text": transcription}
        )


def init_gemini_few_shots() -> None:
    if GEMINI_UPLOADED_FEW_SHOTS:
        return
    for image_path, transcription in get_few_shot_examples():
        img_file = upload_image_gemini(image_path)
        GEMINI_UPLOADED_FEW_SHOTS.append({"image": img_file, "text": transcription})


def build_openai_input(image_path: Path) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, example in enumerate(OPENAI_FEW_SHOTS, start=1):
        content.append({"type": "input_text", "text": f"--- EXAMPLE {index} ---"})
        content.append(
            {
                "type": "input_image",
                "image_url": example["image_url"],
                "detail": OPENAI_IMAGE_DETAIL,
            }
        )
        content.append(
            {"type": "input_text", "text": f"TRANSCRIPTION {index}:\n{example['text']}\n"}
        )

    content.append({"type": "input_text", "text": OCR_PROMPT})
    content.append(
        {
            "type": "input_image",
            "image_url": image_path_to_data_url(image_path),
            "detail": OPENAI_IMAGE_DETAIL,
        }
    )
    return [{"role": "user", "content": content}]


def upload_image_gemini(image_path: Path) -> Any:
    return _call_with_limit(get_gemini_client().files.upload, file=str(image_path))


def create_gemini_cache() -> str:
    if genai_types is None:
        raise RuntimeError("The `google-genai` package is required to use Gemini OCR.")

    contents: list[Any] = []
    for index, example in enumerate(GEMINI_UPLOADED_FEW_SHOTS, start=1):
        contents.append({"text": f"--- EXAMPLE {index} ---"})
        contents.append(example["image"])
        contents.append({"text": f"TRANSCRIPTION {index}:\n{example['text']}\n\n"})

    cache = _call_with_limit(
        get_gemini_client().caches.create,
        model=GEMINI_MODEL_OCR,
        config=genai_types.CreateCachedContentConfig(
            contents=contents,
            system_instruction=SYSTEM_INSTRUCTION,
            ttl="43200s",
        ),
    )
    print(f"Few-shot cache created: {cache.name}: cache time: {cache.expire_time}")
    return cache.name


def init_ocr_context() -> str | None:
    provider = get_provider_name()
    if provider == "openai":
        init_openai_few_shots()
        return None
    init_gemini_few_shots()
    return create_gemini_cache()


def _openai_reason_from_response(response: Any) -> str:
    incomplete = getattr(response, "incomplete_details", None)
    if incomplete is not None:
        reason = getattr(incomplete, "reason", None)
        if reason:
            return str(reason)
    error = getattr(response, "error", None)
    if error is not None:
        return str(error)
    status = getattr(response, "status", None)
    if status:
        return str(status)
    return "unknown"


def ocr_image_openai(image_path: Path) -> tuple[str, str]:
    response = _call_with_limit(
        get_openai_client().responses.create,
        model=OPENAI_MODEL_OCR,
        input=build_openai_input(image_path),
        max_output_tokens=OCR_MAX_OUTPUT_TOKENS,
    )

    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        print(
            "Total Tokens Used: "
            f"{total_tokens} | {input_tokens} input tokens | {output_tokens} output tokens"
        )

    text = getattr(response, "output_text", "") or ""
    reason = "ok" if text else _openai_reason_from_response(response)
    return text, reason


def ocr_image_gemini(image_path: Path, cache_name: str) -> tuple[str, str]:
    if genai_types is None:
        raise RuntimeError("The `google-genai` package is required to use Gemini OCR.")

    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        top_p=1.0,
        top_k=64,
        max_output_tokens=OCR_MAX_OUTPUT_TOKENS,
        response_mime_type="text/plain",
        safety_settings=GEMINI_SAFETY_SETTINGS,
        cached_content=cache_name,
    )
    response = _call_with_limit(
        get_gemini_client().models.generate_content,
        model=GEMINI_MODEL_OCR,
        config=config,
        contents=[upload_image_gemini(image_path), {"text": OCR_PROMPT}],
    )

    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        print(
            "Total Tokens Used: "
            f"{total_tokens} | {prompt_tokens} prompt tokens | {output_tokens} output tokens"
        )

    text = getattr(response, "text", "") or ""
    if not text:
        candidates = getattr(response, "candidates", [])
        reason = (
            str(getattr(candidates[0], "finish_reason", "unknown")) if candidates else "unknown"
        )
    else:
        reason = "ok"
    return text, reason


def ocr_image(image_path: Path, ocr_context: str | None) -> tuple[str, str]:
    if get_provider_name() == "openai":
        return ocr_image_openai(image_path)
    if not ocr_context:
        raise RuntimeError("Gemini OCR requires a cache name.")
    return ocr_image_gemini(image_path, ocr_context)


def ocr_split_image(image_path: Path, output_dir: Path, ocr_context: str | None) -> str:
    """Split image at a whitespace row in the middle third, OCR each half, combine."""
    image = cv2.imread(str(image_path))
    if image is None:
        return ""

    height = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    row_sums = binary.sum(axis=1)

    search_start = height // 3
    search_end = (2 * height) // 3
    split_row = search_start + int(row_sums[search_start:search_end].argmin())

    top_path = output_dir / f"{image_path.name}.top.JPG"
    bottom_path = output_dir / f"{image_path.name}.bottom.JPG"
    cv2.imwrite(str(top_path), image[:split_row, :])
    cv2.imwrite(str(bottom_path), image[split_row:, :])

    try:
        top_text, _ = ocr_image(top_path, ocr_context)
        bottom_text, _ = ocr_image(bottom_path, ocr_context)
        return (top_text + "\n" + bottom_text).strip()
    finally:
        top_path.unlink(missing_ok=True)
        bottom_path.unlink(missing_ok=True)


def process_dir(path: str | Path) -> None:
    provider = get_provider_name()
    print(f"Using OCR provider: {provider}")

    ocr_context = init_ocr_context()
    directory = Path(path)
    output_directory = get_output_directory(directory)
    if output_directory != directory:
        print(f"Writing outputs to: {output_directory}")

    for file_path in sorted(directory.iterdir()):
        if should_skip_input_file(file_path):
            continue

        output_parsed = output_directory / f"{file_path.name}.parsed.txt"
        output_text = output_directory / f"{file_path.name}.txt"

        print(file_path)
        if output_text.exists() and output_text.stat().st_size > 0:
            continue

        raw, reason = ocr_image(file_path, ocr_context)

        if raw == "":
            log_empty(str(file_path), reason)
            converted_image = cv2.imread(str(file_path))
            if converted_image is None:
                continue

            gray = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            temp_path = output_directory / f"{file_path.name}.tresh.JPG"
            cv2.imwrite(str(temp_path), thresh)

            raw, reason = ocr_image(temp_path, ocr_context)
            temp_path.unlink(missing_ok=True)

        if raw == "" and "MAX_TOKENS" in reason:
            raw = ocr_split_image(file_path, output_directory, ocr_context)

        if raw:
            output_parsed.write_text(raw, encoding="utf-8")
            output_text.write_text(strip_tags(raw), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: python gemini_ukr_ocr.py <directory>", file=sys.stderr)
        return 1

    process_dir(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
