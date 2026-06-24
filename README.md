# Ukraine OCR Pipeline

A Python pipeline for transcribing 17th-century Lviv city council documents using Google Gemini 2.5 Pro. The documents contain mixed Latin and Old Polish legal text — council decisions, royal chancery records, contracts, wills, and property records.

## Features

- **Few-shot context caching** — uploads reference exemplars once and caches them for 12 hours to reduce cost and latency
- **Inline language tagging** — the model tags Latin (`[LA]`) and Old Polish (`[PL]`) segments during transcription, improving attention on mixed-language pages
- **Three-tier fallback** — if OCR returns empty, the pipeline automatically retries with a thresholded image, then splits the image horizontally at the nearest whitespace row
- **Dual output** — `.parsed.txt` (tagged, for inspection) and `.txt` (clean, stripped of tags)
- **Rate limiting and retry** — decorrelated jitter backoff with automatic retry on 429/5xx and network errors

## Setup

1. **Install dependencies**

   ```bash
   pip install google-genai opencv-python python-dotenv
   ```

2. **Configure your API key**

   Copy `.env.example` to `.env` and fill in your key:

   ```bash
   cp .env.example .env
   ```

   ```
   GEMINI_API_KEY=your_key_here
   ```

## Usage

```bash
python gemini_ukr_ocr.py <directory>
```

Where `<directory>` contains the `.JPG` images to transcribe. Output files are written alongside the source images:

```
23-2-52/
  001.JPG
  001.JPG.parsed.txt   ← tagged transcription
  001.JPG.txt          ← clean transcription
```

Already-transcribed files (non-empty `.txt`) are skipped automatically.

## Language Tags

The model annotates the transcription inline:

| Tag | Meaning |
|-----|---------|
| `[LA]` | Latin segment |
| `[PL]` | Old Polish segment |
| `[Latin Name: ...]` | Latinized proper name |
| `[Polish Name: ...]` | Polish proper name |

Mixed names are tagged per part: `[Latin Name: Ioannes] [Polish Name: Kowalski]`

## Fallback Pipeline

```
Original image → OCR
      ↓ (empty)
Thresholded image → OCR
      ↓ (empty + MAX_TOKENS)
Split at whitespace row → OCR top half + OCR bottom half → combine
```

Empty responses are logged to `empty_responses.txt` with the finish reason.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required. Your Google Gemini API key. |
| `GEMINI_MODEL_OCR` | `gemini-2.5-pro` | Model used for transcription. |
| `GEMINI_MODEL_CLEAN` | `gemini-2.5-pro` | Model used for post-processing (reserved). |
