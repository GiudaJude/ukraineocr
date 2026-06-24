from pathlib import Path
from types import SimpleNamespace

import pytest

import gemini_ukr_ocr as ocr


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


def test_process_dir_writes_parsed_and_clean_outputs(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")

    monkeypatch.setattr(ocr, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(ocr, "init_ocr_context", lambda: None)
    monkeypatch.setattr(
        ocr,
        "ocr_image",
        lambda image_path, ocr_context: ("[LA] Salve [Polish Name: Kowalski]?", "ok"),
    )

    ocr.process_dir(tmp_path)

    assert (tmp_path / "001.JPG.parsed.txt").read_text(encoding="utf-8") == (
        "[LA] Salve [Polish Name: Kowalski]?"
    )
    assert (tmp_path / "001.JPG.txt").read_text(encoding="utf-8") == "Salve Kowalski"


def test_process_dir_skips_files_with_existing_nonempty_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "001.JPG"
    image_path.write_bytes(b"fake image")
    existing_output = tmp_path / "001.JPG.txt"
    existing_output.write_text("already done", encoding="utf-8")

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
    assert existing_output.read_text(encoding="utf-8") == "already done"


def test_main_requires_exactly_one_directory_argument(capsys) -> None:
    assert ocr.main([]) == 1

    captured = capsys.readouterr()
    assert "Usage: python gemini_ukr_ocr.py <directory>" in captured.err
