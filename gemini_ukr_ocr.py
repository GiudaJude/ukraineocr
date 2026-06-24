import os, sys, time, json, pathlib, re
import time, collections
import random, traceback, requests
import cv2

from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig

# ---------- Configuration ----------
load_dotenv()
MODEL_OCR = os.getenv("GEMINI_MODEL_OCR", "gemini-2.5-pro")
MODEL_CLEAN = os.getenv("GEMINI_MODEL_CLEAN", "gemini-2.5-pro")
API_KEY = os.getenv("GEMINI_API_KEY")

# Old system instruction
'''
SYSTEM_INSTRUCTION = (
    "You are an expert in Ukrainian and Latin languages, especially 17th-century Latin script. "
    "You are an expert in OCR and text extraction from images. "
    "The text in these images contains official council decisions and laws from the Lviv city council in the 17th century, "
    "plus royal chancery/state/church documents and private legal records (contracts, obligations, debts, donations, "
    "inheritance distributions, wills, sales agreements, hypothecs, property divisions)."
)
'''
# New
SYSTEM_INSTRUCTION = (
    "ROLE: You are an expert peleographer specializing in 16th and 17th century Lviv council records and Latin/Old Polish legal scripts."
    "TASK: Transcribe the provided image into a full-text format."
    "TRANSCRIPTION RULES:"
    "1. Expand Abbreviations."
    "You will encounter many scribal abbreviations (such as marks above words like 'Leopolien')."
    "Do not transcribe these literally. Expand them into the gramatically correct Latin form (e.g. Leopoliensis, Leopolitanus) based on the surrounding sentence structure."
    "2. Grounding:"
    "Strictly follow the expension logic and formatting style demonstrated in the few shot exemplar."
    "3. Contextual Accuracy"
    "In the case of smears and smudges, use the context of the surrounded words and letters"
    "4. Output Only"
    "Provide only the final transcribed text. Do not include any thoughts in your output unless asked"
    "5. NO EMPTY RESPONSES: You MUST output a transcription. Even if the page is highly degraded, provide your best attempt. Never return an empty string."
)

SAFETY = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

client = genai.Client(api_key=API_KEY)

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

UPLOADED_FEW_SHOTS = []

def log_empty(img_path: str, reason: str, log_path: str = "empty_responses.txt"):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{img_path} reason for emptiness: {reason}\n")

def strip_tags(text: str) -> str:
    text = re.sub(r'\[(LA|PL)\]\s*', '', text)
    text = re.sub(r'\[Latin Name:\s*(.*?)\]', r'\1', text)
    text = re.sub(r'\[Polish Name:\s*(.*?)\]', r'\1', text)
    text = text.replace('?', '')
    return text.strip()

# ------- Global rate limiter: 2 requests per 60s -------
class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: int):
        self.max_calls = max_calls
        self.per = per_seconds
        self.history = collections.deque()

    def wait(self):
        now = time.time()
        while self.history and now - self.history[0] >= self.per:
            self.history.popleft()
        if len(self.history) >= self.max_calls:
            sleep_for = self.per - (now - self.history[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.history.append(time.time())

limiter = RateLimiter(max_calls=2, per_seconds=60)

RETRIABLE_STATUS = {429, 500, 502, 503, 504}

def _status_from_exc(e) -> int | None:
    for attr in ("status", "code", "http_status", "http_code"):
        v = getattr(e, attr, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    resp = getattr(e, "response", None) or getattr(e, "http_response", None)
    if resp is not None:
        status_code = getattr(resp, "status_code", None)
        if status_code is not None:
            try:
                return int(status_code)
            except Exception:
                pass
    s = str(e)
    for code in RETRIABLE_STATUS:
        if str(code) in s:
            return code
    return None

def _parse_retry_after_seconds(err) -> float | None:
    resp = getattr(err, "response", None) or getattr(err, "http_response", None)
    headers = getattr(resp, "headers", None)
    if isinstance(headers, dict):
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except Exception:
                return None
    return None

def _is_network_error(e: Exception) -> bool:
    return isinstance(e, (
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError
    ))

def _call_with_limit(fn, *args, **kwargs):
    """Rate-limit + robust retries for any Gemini SDK call."""
    attempt = 0
    base = 2.0
    cap = 90.0
    sleep = base

    while True:
        limiter.wait()
        try:
            return fn(*args, **kwargs)

        except Exception as e:
            attempt += 1
            status = _status_from_exc(e)
            is_retriable = (
                _is_network_error(e)
                or (status in RETRIABLE_STATUS)
                or ("rate" in str(e).lower())
                or ("overloaded" in str(e).lower())
                or ("temporarily unavailable" in str(e).lower())
            )

            print(f"[retry {attempt}] {type(e).__name__} {status or ''} ? {e}")
            tb = traceback.format_exc(limit=1).strip()
            if tb:
                print(tb)

            if not is_retriable and attempt >= 1:
                raise

            ra = _parse_retry_after_seconds(e)
            if ra is not None:
                time.sleep(max(1.0, ra))
                continue

            upper = min(cap, sleep * 3.0)
            sleep = random.uniform(base, max(base, upper))
            time.sleep(sleep)

            if attempt >= 8:
                raise

# ------------------ Upload once, reuse ------------------
def upload_image(path: str):
    return _call_with_limit(client.files.upload, file=path)

def init_few_shots():
    if UPLOADED_FEW_SHOTS:
        return

    for item in FEW_SHOT_DATA:
        img_path = item["image_path"]
        txt_path = item["text_path"]

        if os.path.exists(img_path) and os.path.exists(txt_path):
            img_file = upload_image(img_path)
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_file = f.read().strip()
            UPLOADED_FEW_SHOTS.append({"image": img_file, "text": txt_file})
        else:
            print(f"WARNING: Missing {img_path} or {txt_path}. Skipping")

def fewshot_cache():
    contents = []
    for i, example in enumerate(UPLOADED_FEW_SHOTS, start=1):
        contents.append({"text": f"--- EXAMPLE {i} ---"})
        contents.append(example["image"])
        contents.append({"text": f"TRANSCRIPTION {i}:\n{example['text']}\n\n"})

    cache = _call_with_limit(
        client.caches.create,
        model=MODEL_OCR,
        config=types.CreateCachedContentConfig(
            contents=contents,
            system_instruction=SYSTEM_INSTRUCTION,
            ttl="43200s",
        ),
    )
    print(f"Few-shot cache created: {cache.name}: cache time: {cache.expire_time}")
    return cache.name

def ocr_image_from_file(img_file, cache_name) -> tuple[str, str]:
    """Transcribe image with inline language tagging."""
    cfg = GenerateContentConfig(
        temperature=0.0, top_p=1.0, top_k=64,
        max_output_tokens=32768,
        response_mime_type="text/plain",
        safety_settings=SAFETY,
        cached_content=cache_name,
    )
    prompt = (
        "Carefully transcribe all visible, forward-facing text. "
        "Do not translate or paraphrase. Preserve original line breaks and obvious hyphenations.\n\n"
        "As you transcribe, apply these tags:\n"
        "- Prefix each Latin segment with [LA]\n"
        "- Prefix each Old Polish segment with [PL]\n"
        "- Wrap Latin proper names (Latinized form) as [Latin Name: ...]\n"
        "- Wrap Polish proper names (Polish form) as [Polish Name: ...]\n"
        "- For names with a Latin first name and Polish surname (or vice versa), "
        "tag each part separately: [Latin Name: Ioannes] [Polish Name: Kowalski]\n"
        "Output only the tagged transcription."
    )
    resp = _call_with_limit(
        client.models.generate_content,
        model=MODEL_OCR,
        config=cfg,
        contents=[img_file, {"text": prompt}]
    )

    if resp.usage_metadata:
        prompt_tokens = resp.usage_metadata.prompt_token_count
        out_tokens = resp.usage_metadata.candidates_token_count
        total_tokens = resp.usage_metadata.total_token_count
        print(f"Total Tokens Used: {total_tokens}| {prompt_tokens} prompt tokens | {out_tokens} output tokens")

    text = getattr(resp, "text", "") or ""
    if not text:
        candidates = getattr(resp, "candidates", [])
        reason = str(getattr(candidates[0], "finish_reason", "unknown")) if candidates else "unknown"
    else:
        reason = "ok"
    return text, reason

def ocr_split_image(f: pathlib.Path, p: pathlib.Path, cache_name: str) -> str:
    """Split image at a whitespace row in the middle third, OCR each half, combine."""
    img = cv2.imread(str(f))
    if img is None:
        return ""
    h = img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    row_sums = binary.sum(axis=1)

    search_start = h // 3
    search_end = (2 * h) // 3
    split_row = search_start + int(row_sums[search_start:search_end].argmin())

    top_path = p / f"{f.name}.top.JPG"
    bottom_path = p / f"{f.name}.bottom.JPG"
    cv2.imwrite(str(top_path), img[:split_row, :])
    cv2.imwrite(str(bottom_path), img[split_row:, :])

    try:
        top_file = upload_image(str(top_path))
        top_text, _ = ocr_image_from_file(top_file, cache_name)

        bottom_file = upload_image(str(bottom_path))
        bottom_text, _ = ocr_image_from_file(bottom_file, cache_name)

        return (top_text + "\n" + bottom_text).strip()
    finally:
        top_path.unlink(missing_ok=True)
        bottom_path.unlink(missing_ok=True)

def process_dir(path: str):
    init_few_shots()
    cache_name = fewshot_cache()
    p = pathlib.Path(path)
    for f in sorted(p.iterdir()):
        if f.name.startswith('.'):
            continue
        if f.is_dir():
            continue
        if f.suffix.lower() in {".txt", ".json"}:
            continue
        if ".tresh." in f.name or ".top." in f.name or ".bottom." in f.name:
            continue

        base = f.name
        out_parsed = p / (base + ".parsed.txt")
        out_txt = p / (base + ".txt")

        print(f)
        if out_txt.exists() and out_txt.stat().st_size > 0:
            continue

        # 1) Upload and OCR
        img_file = upload_image(str(f))
        raw, reason = ocr_image_from_file(img_file, cache_name)

        # 2) Fallback: threshold (simpler image = fewer thinking tokens)
        if raw == "":
            log_empty(str(f), reason)
            converted_image = cv2.imread(str(f))
            if converted_image is None:
                continue
            gray = cv2.cvtColor(converted_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            temp_path = p / f"{base}.tresh.JPG"
            cv2.imwrite(str(temp_path), thresh)

            img_file = upload_image(str(temp_path))
            raw, reason = ocr_image_from_file(img_file, cache_name)
            temp_path.unlink(missing_ok=True)

        # 3) Fallback: split at whitespace row if still hitting MAX_TOKENS
        if raw == "" and "MAX_TOKENS" in reason:
            raw = ocr_split_image(f, p, cache_name)

        if raw:
            out_parsed.write_text(raw, encoding="utf-8")
            out_txt.write_text(strip_tags(raw), encoding="utf-8")


if __name__ == "__main__":
    import sys
    process_dir(sys.argv[1])
