import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as ocr


def _make_word(**overrides) -> "ocr.WordClassification":
    fields = {
        "word": "Salve",
        "language": "Latin",
        "language_confidence_score": 0.95,
        "language_confidence_reasoning": "Common Latin greeting.",
        "word_declension": None,
        "word_type": "other",
        "line_number": 1,
        "transcription_confidence_score": 0.9,
        "transcription_confidence_reasoning": "Clearly legible.",
    }
    fields.update(overrides)
    return ocr.WordClassification(**fields)


def test_get_provider_name_prefers_openai_when_both_keys_exist(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(ocr, "GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(ocr, "OpenAI", object())
    monkeypatch.setattr(ocr, "genai", object())

    assert ocr.get_provider_name() == "openai"


def test_get_provider_name_falls_back_to_gemini_when_openai_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(ocr, "GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(ocr, "OpenAI", None)
    monkeypatch.setattr(ocr, "genai", object())

    assert ocr.get_provider_name() == "gemini"


def test_get_provider_name_requires_a_configured_key(monkeypatch) -> None:
    monkeypatch.setattr(ocr, "OPENAI_API_KEY", None)
    monkeypatch.setattr(ocr, "GEMINI_API_KEY", None)

    with pytest.raises(RuntimeError, match="No API key is configured"):
        ocr.get_provider_name()


def test_strip_tags_removes_markup_and_question_marks() -> None:
    tagged = (
        "[LA] Dominus\n"
        "[PL] Jan\n"
        "[Latin Name: Ioannes] [Polish Name: Kowalski]?"
    )

    assert ocr.strip_tags(tagged) == "Dominus\nJan\nIoannes Kowalski"


def test_image_path_to_data_url_encodes_jpeg_images(tmp_path: Path) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"test-bytes")

    result = ocr.image_path_to_data_url(image_path)

    assert result.startswith("data:image/jpeg;base64,")


def test_build_openai_input_includes_configured_detail(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"test-bytes")

    monkeypatch.setattr(ocr, "OPENAI_FEW_SHOTS", [])
    monkeypatch.setattr(ocr, "OPENAI_IMAGE_DETAIL", "low")

    payload = ocr.build_openai_input(image_path)
    image_items = [item for item in payload[0]["content"] if item["type"] == "input_image"]

    assert image_items
    assert all(item["detail"] == "low" for item in image_items)


def test_status_from_exc_reads_explicit_status_attribute() -> None:
    error = RuntimeError("boom")
    error.status = "503"  # type: ignore[attr-defined]

    assert ocr._status_from_exc(error) == 503


def test_status_from_exc_falls_back_to_message_text() -> None:
    error = RuntimeError("temporary failure: 504 gateway timeout")

    assert ocr._status_from_exc(error) == 504


def test_parse_retry_after_seconds_uses_response_headers() -> None:
    error = RuntimeError("retry later")
    error.response = SimpleNamespace(headers={"Retry-After": "12.5"})  # type: ignore[attr-defined]

    assert ocr._parse_retry_after_seconds(error) == 12.5


def test_should_skip_input_file_handles_hidden_dirs_and_intermediate_outputs(
    tmp_path: Path,
) -> None:
    hidden = tmp_path / ".hidden.JPG"
    hidden.write_bytes(b"x")
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    intermediate = tmp_path / "page.tresh.JPG"
    intermediate.write_bytes(b"x")
    json_file = tmp_path / "page.json"
    json_file.write_text("{}", encoding="utf-8")
    page = tmp_path / "page.JPG"
    page.write_bytes(b"x")

    assert ocr.should_skip_input_file(hidden) is True
    assert ocr.should_skip_input_file(nested_dir) is True
    assert ocr.should_skip_input_file(intermediate) is True
    assert ocr.should_skip_input_file(json_file) is True
    assert ocr.should_skip_input_file(page) is False


def test_get_output_directory_defaults_to_input_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ocr, "OCR_OUTPUT_ROOT", None)

    assert ocr.get_output_directory(tmp_path) == tmp_path


def test_get_output_directory_uses_output_root(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "sample_data" / "17-2-52"
    source_dir.mkdir(parents=True)
    output_root = tmp_path / "ocr_runs" / "run-1"
    monkeypatch.setattr(ocr, "OCR_OUTPUT_ROOT", str(output_root))

    output_dir = ocr.get_output_directory(source_dir)

    assert output_dir == output_root / source_dir.name
    assert output_dir.exists()


def test_process_dir_writes_parsed_clean_and_words_outputs(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")
    word = _make_word()

    monkeypatch.setattr(ocr, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(ocr, "init_ocr_context", lambda: None)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda image_path, ocr_context: (
            "[LA] Salve [Polish Name: Kowalski]?",
            [word],
            "ok",
        ),
    )

    ocr.process_dir(tmp_path)

    assert (tmp_path / "001.JPG.parsed.txt").read_text(encoding="utf-8") == (
        "[LA] Salve [Polish Name: Kowalski]?"
    )
    assert (tmp_path / "001.JPG.txt").read_text(encoding="utf-8") == "Salve Kowalski"

    words_payload = json.loads((tmp_path / "001.JPG.words.json").read_text(encoding="utf-8"))
    assert words_payload["source_image"] == "001.JPG"
    assert words_payload["page_number"] == 1
    assert words_payload["words"] == [word.model_dump()]


def test_process_dir_skips_files_with_existing_words_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")
    existing_words = tmp_path / "001.JPG.words.json"
    existing_words.write_text('{"words": []}', encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(ocr, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(ocr, "init_ocr_context", lambda: None)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda image_path, ocr_context: calls.append(str(image_path)),
    )

    ocr.process_dir(tmp_path)

    assert calls == []
    assert existing_words.read_text(encoding="utf-8") == '{"words": []}'


def test_process_dir_reprocesses_when_only_legacy_txt_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pre-existing .parsed.txt/.txt from before this feature must not block
    reprocessing -- only .words.json existence should skip an image."""
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")
    (tmp_path / "001.JPG.txt").write_text("already done", encoding="utf-8")
    (tmp_path / "001.JPG.parsed.txt").write_text("[LA] already done", encoding="utf-8")
    word = _make_word(word="Novum")

    monkeypatch.setattr(ocr, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(ocr, "init_ocr_context", lambda: None)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda image_path, ocr_context: ("[LA] Novum", [word], "ok"),
    )

    ocr.process_dir(tmp_path)

    assert (tmp_path / "001.JPG.txt").read_text(encoding="utf-8") == "Novum"
    assert (tmp_path / "001.JPG.words.json").exists()


def test_process_dir_logs_warning_and_skips_words_json_when_classification_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")

    monkeypatch.setattr(ocr, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(ocr, "init_ocr_context", lambda: None)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda image_path, ocr_context: ("[LA] Salve", None, "invalid_schema: boom"),
    )

    ocr.process_dir(tmp_path)

    assert (tmp_path / "001.JPG.txt").read_text(encoding="utf-8") == "Salve"
    assert not (tmp_path / "001.JPG.words.json").exists()


def test_parse_page_transcription_accepts_valid_payload() -> None:
    payload = json.dumps(
        {
            "transcription": "[LA] Salve",
            "words": [
                {
                    "word": "Salve",
                    "language": "Latin",
                    "language_confidence_score": 0.95,
                    "language_confidence_reasoning": "Common Latin greeting.",
                    "word_declension": None,
                    "word_type": "other",
                    "line_number": 1,
                    "transcription_confidence_score": 0.9,
                    "transcription_confidence_reasoning": "Clearly legible.",
                }
            ],
        }
    )

    page = ocr.parse_page_transcription(payload)

    assert page.transcription == "[LA] Salve"
    assert page.words[0].word == "Salve"
    assert page.words[0].language == "Latin"


def test_parse_page_transcription_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        ocr.parse_page_transcription("not json")


def test_parse_page_transcription_rejects_schema_violation() -> None:
    payload = json.dumps({"transcription": "text", "words": [{"word": "x"}]})

    with pytest.raises(ocr.ValidationError):
        ocr.parse_page_transcription(payload)


def test_salvage_transcription_recovers_bare_transcription(tmp_path: Path) -> None:
    raw_output = json.dumps({"transcription": "[LA] Salve", "words": [{"word": "x"}]})

    text, words, reason = ocr._salvage_transcription(
        raw_output, tmp_path / "001.JPG", ValueError("bad words entry")
    )

    assert text == "[LA] Salve"
    assert words is None
    assert reason == "ok"


def test_salvage_transcription_gives_up_on_total_garbage(tmp_path: Path) -> None:
    text, words, reason = ocr._salvage_transcription(
        "not json at all", tmp_path / "001.JPG", ValueError("boom")
    )

    assert text == ""
    assert words is None
    assert reason.startswith("invalid_schema")


def test_main_requires_exactly_one_directory_argument(capsys) -> None:
    assert ocr.main([]) == 1

    captured = capsys.readouterr()
    assert "Usage: python gemini_ukr_ocr.py <directory>" in captured.err
