# Ukraine OCR Pipeline

A Python pipeline for transcribing 17th-century Lviv city council documents with OpenAI or Google Gemini. The documents contain mixed Latin and Old Polish legal text — council decisions, royal chancery records, contracts, wills, and property records.

## Features

- **Provider-aware OCR** — uses OpenAI by default when `OPENAI_API_KEY` is present, otherwise falls back to Gemini when `GEMINI_API_KEY` is set
- **Few-shot exemplars** — reuses reference exemplars for transcription style grounding; Gemini caches them for 12 hours to reduce cost and latency
- **Inline language tagging** — the model tags Latin (`[LA]`) and Old Polish (`[PL]`) segments during transcription, improving attention on mixed-language pages
- **Three-tier fallback** — if OCR returns empty, the pipeline automatically retries with a thresholded image, then splits the image horizontally at the nearest whitespace row
- **Triple output** — `.parsed.txt` (tagged, for inspection), `.txt` (clean, stripped of tags), and `.words.json` (per-word language/declension/type breakdown with confidence scores)
- **Rate limiting and retry** — decorrelated jitter backoff with automatic retry on 429/5xx and network errors

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Install dev tooling**

   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Configure your API key**

   Copy `.env.example` to `.env` and fill in your key:

   ```bash
   cp .env.example .env
   ```

   ```
   OPENAI_API_KEY=your_key_here
   ```

   Or, if you want to use Gemini instead:

   ```
   GEMINI_API_KEY=your_key_here
   ```

   If both keys are set, the script uses OpenAI by default.

## Usage

```bash
python gemini_ukr_ocr.py <directory>
```

Where `<directory>` contains the `.JPG` images to transcribe. By default, output files are written alongside the source images:

```
23-2-52/
  001.JPG
  001.JPG.parsed.txt   ← tagged transcription
  001.JPG.txt          ← clean transcription
  001.JPG.words.json   ← per-word language/declension/type breakdown
```

Already-transcribed files are skipped based on `.words.json` alone (non-empty).
`.parsed.txt`/`.txt` existing on their own does **not** skip an image — running
the pipeline again over a directory that predates the `.words.json` output will
regenerate and overwrite its `.parsed.txt`/`.txt` too.

For comparison runs, it is cleaner to write outputs into a separate run folder:

```bash
OCR_OUTPUT_ROOT=ocr_runs/2026-06-24-smoke-openai python gemini_ukr_ocr.py sample_data/17-2-52
```

That produces a mirrored layout like:

```text
ocr_runs/2026-06-24-smoke-openai/
  sample_data/
    17-2-52/
      0061.JPG.parsed.txt
      0061.JPG.txt
```

## Development

Run the linter:

```bash
ruff check .
```

Run the test suite:

```bash
pytest
```

Validate the local sample corpus without calling the OCR API:

```bash
pytest tests/test_sample_data_integration.py -k readable
```

Extract canonical entities and relationship triples from a single document's OCR text
(provider-aware: OpenAI if OPENAI_API_KEY is set, else Gemini):

```bash
python entity_graph.py ocr_runs/smoke-live-20260624-120351/17-2-52/0061.JPG.txt
```

That writes a strict-schema JSON file next to the input text by default:

```text
0061.JPG.txt.entity_graph.json
```

### Build the corpus-wide entity registry

`entity_graph.py` canonicalizes entities within a single page, it has no memory
of any other page. `entity_registry.py` incrementally merges every page's extraction
into one persistent, corpus-wide registry, so that name/place variants referring to
the same real-world person or place — e.g. a Latinized name and its vernacular form,
or a place's historical Latin/Polish/Ukrainian names — accumulate into a single
canonical record instead of scattering across hundreds of disconnected per-page files.

```bash
python entity_registry.py 23-2-52          # one batch directory (recurses into subfolders)
python entity_registry.py .                # whole corpus
python entity_registry.py 23-2-52 --reprocess   # force re-merge if a page's .txt changed
```

This reads every `*.JPG.txt` file under the given directory and writes/updates two files:

- `entity_registry/registry.json` — tool-managed. One entry per real-world entity, with
  every alias/spelling seen so far and which document(s) it came from. **Commit this to
  git** — it's curated derived data, not a disposable run artifact.
- `entity_registry/lookup.json` — hand-curated by you, never written by the tool. Seed it
  with known variant groups, e.g.:
  ```json
  [
    {
      "entity_type": "location",
      "canonical_name": "Lwów",
      "aliases": ["Leopolis", "Leopoliensis", "Lviv", "Lwów", "Львів", "Lemberg"],
      "notes": "Historical Latin/Polish/Ukrainian/German names for the same city."
    }
  ]
  ```
  Also **commit this to git** — start it as `[]` if you have nothing to seed yet.

Already-processed, unchanged documents are skipped on re-runs (tracked by content hash),
so it's safe/cheap to re-run after OCR-ing more pages — no repeat API calls or cost for
pages already merged in.

Run a live OCR smoke test on a small subset of `sample_data`:

```bash
OCR_OUTPUT_ROOT=ocr_runs/smoke-live RUN_SAMPLE_OCR_LIVE=1 SAMPLE_OCR_LIMIT=2 OCR_FEW_SHOT_LIMIT=1 OCR_MAX_OUTPUT_TOKENS=4000 OPENAI_IMAGE_DETAIL=low pytest -m integration -s
```

This integration test is opt-in because it makes real API calls and can incur cost.

## Language Tags

The model annotates the transcription inline:

| Tag | Meaning |
|-----|---------|
| `[LA]` | Latin segment |
| `[PL]` | Old Polish segment |
| `[Latin Name: ...]` | Latinized proper name |
| `[Polish Name: ...]` | Polish proper name |

Mixed names are tagged per part: `[Latin Name: Ioannes] [Polish Name: Kowalski]`

## Per-Word Breakdown (`.words.json`)

Alongside the tagged/clean text, the OCR call also returns a structured
per-word breakdown, written to `<image>.words.json`:

```json
{
  "source_image": "0061.JPG",
  "page_number": 61,
  "words": [
    {
      "word": "Leopoliensis",
      "language": "Latin",
      "language_confidence_score": 0.97,
      "language_confidence_reasoning": "Standard Latinized place-adjective.",
      "word_declension": null,
      "word_type": "location",
      "line_number": 1,
      "transcription_confidence_score": 0.6,
      "transcription_confidence_reasoning": "Ink is smudged around the middle syllable."
    }
  ]
}
```

- `language` — the model's own judgment (`Latin` / `Polish` / `Ukrainian` / `other`),
  independently of the `[LA]`/`[PL]` tag in the transcription.
- `word_declension` — grammatical case/number for Polish words (e.g. `"genitive singular"`);
  `null` otherwise.
- `word_type` — `name` / `location` / `verb` / `subject` / `object` / `other`.
- `transcription_confidence_score`/`_reasoning` — a *separate* confidence score for
  whether the word's letters were read correctly (legibility), independent of the
  language classification.
- `page_number` is derived deterministically from the filename, not model-generated.

This is produced by the same OCR call as the transcription (via a strict
JSON-schema response on OpenAI, or a `response_schema` on Gemini), not a second
API call, so it's always consistent with `.parsed.txt`. If the word list fails to
parse/validate, the transcription is still kept and a warning is logged — the
image is retried automatically on the next run since `.words.json` won't exist yet.

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
| `OPENAI_API_KEY` | — | Preferred. If present, OpenAI is used for OCR. |
| `OPENAI_MODEL_OCR` | `gpt-5.4-mini` | Default OpenAI OCR model for faster, lower-cost runs. |
| `OPENAI_IMAGE_DETAIL` | `low` | OpenAI vision detail level. `low` is fastest/cheapest; raise to `high` if OCR quality needs it. |
| `OPENAI_MODEL_NER` | `gpt-5.4-mini` | Default OpenAI model for entity/relationship extraction. |
| `OPENAI_NER_MAX_OUTPUT_TOKENS` | `4000` | Caps entity extraction response size. |
| `OCR_FEW_SHOT_LIMIT` | `8` | Limits how many exemplars are included in OCR requests. Lower this for faster/cheaper smoke tests. |
| `OCR_MAX_OUTPUT_TOKENS` | `32768` | Caps OCR response size. Lower this for smoke tests. |
| `OCR_OUTPUT_ROOT` | — | Optional directory for writing OCR outputs separately from the source images. |
| `GEMINI_API_KEY` | — | Fallback key. Used when no OpenAI key is configured. |
| `GEMINI_MODEL_OCR` | `gemini-2.5-pro` | Gemini OCR model. |
| `GEMINI_MODEL_NER` | `gemini-3.7-flash` | Default Gemini model for entity/relationship extraction. |
| `GEMINI_NER_MAX_OUTPUT_TOKENS` | `4000` | Caps entity extraction response size on Gemini. |
| `ENTITY_FUZZY_MAX_DISTANCE_RATIO` | `0.15` | Edit-distance safety net for merging location/organization names in the registry (never applied to person names). |
| `ENTITY_CONTEXT_LIMIT` | `200` | Max number of known registry entities included as context in each extraction prompt. |

`OCR_MAX_OUTPUT_TOKENS` now bounds the transcription *and* the per-word JSON
combined, so pages with many words are more likely to hit the limit than before
— watch for `MAX_TOKENS`-triggered split-image fallbacks in the logs and raise
it if truncations spike.

## Known Limitations

- **Location canonicalization is mostly automatic.** The model can usually link
  `Leopolis`/`Lwów`/`Lviv`/`Lemberg` on its own from general world knowledge, and
  `entity_registry.py` reinforces this by feeding each new page a summary of
  already-known canonical names to reuse.
- **Person canonicalization is not automatic across documents.** Recognizing that
  `Ioannes Kowalski` on one page and `Jan Kowalski` on another are the same person
  usually requires either a contextual clue in the text itself (same patronymic,
  occupation, address) or a manual entry in `entity_registry/lookup.json`. Expect to
  periodically skim `entity_registry/registry.json` and add lookup entries by hand
  for people the tool didn't merge on its own.
