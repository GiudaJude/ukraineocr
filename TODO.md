# TODO

## Verify the Polish-skip change (main.py prompts)
- [ ] Run on `0003.JPG` (Latin list with Polish surnames). Expect the surnames (Miesskowski, Gulinszki, Wolfowicz, Przedziecki, Madrowicz) tagged `[PL]` and `Doctore`/`Notario` kept `[LA]`.
- [ ] Run on `0004.JPG` (blank page with mirrored show-through). Expect an empty `transcription` and `words`.
- [ ] Run on a genuinely faded page, to check rule 7 doesn't drop real text.
- [ ] Run on a fully Polish page (was `Reference4`). Expect it to be skipped.
- [ ] Spot-check whether the model skips borderline Old Polish/Latin mixed passages that it should keep.

## Empty-page retry in `process_dir` (main.py, ~line 716)
- [ ] The retry on an empty result uses an Otsu-thresholded image. On show-through pages this turns the faint mirrored ink solid black, which makes the model more likely to transcribe or invent text. Options:
  - skip the retry when the model returns empty
  - only retry if the image has enough dark pixels
  - only retry on `MAX_TOKENS` or `SAFETY` finish reasons
- [ ] Decide how empty pages should be recorded, e.g. write an empty `.txt` and `.words.json` so the page isn't reprocessed.
- [ ] `empty_responses.txt` will now also log genuinely blank pages. Consider a separate reason label.

## Exemplars
- [ ] `Reference4` has been removed from `FEW_SHOT_DATA`, but the files are still in `exemplars/`. Decide whether to keep or delete them.
- [ ] `Reference2.txt` spells the surname `Mozanc`, but the `OCR_PROMPT` example uses `Monzanc`. Check the manuscript and make them match.
- [ ] `Abbreviation1.txt` and `Abbreviation2.txt` are near duplicates. Drop one to save tokens on every call.
- [ ] The exemplars have no `[LA]`/`[PL]` tags, but the prompt asks for them. Consider tagging the surnames in `Reference1–3.txt`.
- [ ] Add a short exemplar for an empty or show-through page, to reinforce rule 8.

## Downstream
- [ ] Check that `strip_tags`, the entity registry and the graph code cope with pages that have no words, and with `words` containing only surnames.
- [ ] Re-run existing pages. Output files with a non-empty `.words.json` are skipped (`process_dir`, ~line 711), so old outputs won't pick up the new prompt until they're deleted.

## Config and docs cleanup
- [ ] `.env.example` still has `GEMINI_MODEL_OCR=gemini-2.5-pro`, but the code default is `gemini-3.7-flash`. Decide which one you want and make them agree.
- [ ] `.env.example` has `CONSISTENCY_RARE_MA=2`, but `post_tokenization.py` reads `CONSISTENCY_RARE_MAX`. This looks like a typo, so the setting is currently ignored.
- [ ] `.env.example` is missing most of the variables in the README's env table (`OCR_FEW_SHOT_LIMIT`, `OCR_MAX_OUTPUT_TOKENS`, `OCR_OUTPUT_ROOT`, the NER, registry and graph variables).
- [ ] `post_tokenization.py` and its variables (`CONSISTENCY_RARE_MAX`, `CONSISTENCY_COMMON_MIN`, `CONSISTENCY_MAX_DISTANCE_RATIO`) are not documented in the README. Add a section and put them in the env table.
- [ ] `likely_misreads.csv`, `unmatched_rarities.csv` and `mulciber/` are not mentioned anywhere in the README. Document them, or move them out of the repo root.
- [ ] `README.md` has no example of the new behaviour on a real page. Add a before/after snippet once you've run `0003.JPG`.