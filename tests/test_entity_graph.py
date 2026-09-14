from pathlib import Path

import pytest

import entity_graph as graph


def test_get_openai_client_reads_api_key_from_environment(monkeypatch) -> None:
    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key

    monkeypatch.setattr(graph, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(graph, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(graph, "OPENAI_CLIENT", None)

    client = graph.get_openai_client()

    assert client.api_key == "test-key"


def test_get_openai_client_raises_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(graph, "OPENAI_API_KEY", None)
    monkeypatch.setattr(graph, "OPENAI_CLIENT", None)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        graph.get_openai_client()


def test_get_gemini_client_reads_api_key_from_environment(monkeypatch) -> None:
    class FakeGeminiClient:
        def __init__(self, api_key: str):
            self.api_key = api_key

    class FakeGenaiModule:
        Client = FakeGeminiClient

    monkeypatch.setattr(graph, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(graph, "genai", FakeGenaiModule)
    monkeypatch.setattr(graph, "GEMINI_CLIENT", None)

    client = graph.get_gemini_client()

    assert client.api_key == "test-gemini-key"


def test_get_gemini_client_raises_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.setattr(graph, "GEMINI_API_KEY", None)
    monkeypatch.setattr(graph, "GEMINI_CLIENT", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        graph.get_gemini_client()


def test_get_provider_name_prefers_openai_when_both_keys_set(monkeypatch) -> None:
    monkeypatch.setattr(graph, "OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(graph, "GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(graph, "OpenAI", object())
    monkeypatch.setattr(graph, "genai", object())

    assert graph.get_provider_name() == "openai"


def test_get_provider_name_falls_back_to_gemini(monkeypatch) -> None:
    monkeypatch.setattr(graph, "OPENAI_API_KEY", None)
    monkeypatch.setattr(graph, "GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(graph, "genai", object())

    assert graph.get_provider_name() == "gemini"


def test_get_provider_name_raises_when_no_key_configured(monkeypatch) -> None:
    monkeypatch.setattr(graph, "OPENAI_API_KEY", None)
    monkeypatch.setattr(graph, "GEMINI_API_KEY", None)

    with pytest.raises(RuntimeError, match="No API key is configured"):
        graph.get_provider_name()


def test_get_provider_name_raises_when_key_set_but_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(graph, "OPENAI_API_KEY", "openai-key")
    monkeypatch.setattr(graph, "GEMINI_API_KEY", None)
    monkeypatch.setattr(graph, "OpenAI", None)

    with pytest.raises(RuntimeError, match="openai.*is not installed"):
        graph.get_provider_name()


def test_default_output_path_appends_entity_graph_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "page.txt"
    input_path.write_text("hello", encoding="utf-8")

    assert graph.default_output_path(input_path) == tmp_path / "page.txt.entity_graph.json"


def test_build_prompt_without_known_entities_context_has_no_extra_section() -> None:
    prompt = graph.build_prompt("some document text", "page.txt")

    assert "Known entities already established" not in prompt
    assert "Document ID: page.txt" in prompt
    assert "some document text" in prompt


def test_build_prompt_with_known_entities_context_includes_it() -> None:
    prompt = graph.build_prompt("some document text", "page.txt", known_entities_context="LOC-0001: Lwow (aka Leopolis)")

    assert "Known entities already established" in prompt
    assert "LOC-0001: Lwow (aka Leopolis)" in prompt


def test_validate_entity_graph_json_accepts_valid_payload() -> None:
    raw_json = """
    {
      "document_id": "page.txt",
      "entities": [
        {
          "entity_id": "E1",
          "entity_type": "person",
          "canonical_name": "Gabriel Banas",
          "mention_texts": ["Gabriel Banas"],
          "justification": "Named explicitly in the document."
        },
        {
          "entity_id": "E2",
          "entity_type": "location",
          "canonical_name": "Lviv",
          "mention_texts": ["Leopoliensis"],
          "justification": "Leopoliensis refers to Lviv."
        }
      ],
      "relationships": [
        {
          "head_entity_id": "E1",
          "relation": "located_in",
          "tail_entity_id": "E2",
          "evidence": "Gabriel Banas is described as associated with Leopoliensis."
        }
      ]
    }
    """

    parsed = graph.validate_entity_graph_json(raw_json)

    assert parsed.document_id == "page.txt"
    assert len(parsed.entities) == 2
    assert parsed.relationships[0].head_entity_id == "E1"


def test_validate_entity_graph_json_rejects_missing_entity_references() -> None:
    raw_json = """
    {
      "document_id": "page.txt",
      "entities": [],
      "relationships": [
        {
          "head_entity_id": "E1",
          "relation": "related_to",
          "tail_entity_id": "E2",
          "evidence": "Mentioned together."
        }
      ]
    }
    """

    with pytest.raises(RuntimeError, match="missing from entities"):
        graph.validate_entity_graph_json(raw_json)


def test_extract_entity_graph_openai_passes_known_entities_context(monkeypatch) -> None:
    """Regression test: the OpenAI path once silently dropped known_entities_context instead of threading it into build_prompt"""
    captured_contexts: list[str] = []

    def fake_build_prompt(document_text, document_id, known_entities_context=""):
        captured_contexts.append(known_entities_context)
        return "irrelevant prompt text"

    class FakeResponse:
        output_text = '{"document_id": "page.txt", "entities": [], "relationships": []}'

    class FakeResponses:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeOpenAIClient:
        responses = FakeResponses()

    monkeypatch.setattr(graph, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(graph, "get_openai_client", lambda: FakeOpenAIClient())

    graph.extract_entity_graph_openai("document text", "page.txt", known_entities_context="LOC-0001: Lwow")

    assert captured_contexts == ["LOC-0001: Lwow"]


def test_extract_entity_graph_gemini_passes_response_schema_and_system_instruction(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        text = '{"document_id": "page.txt", "entities": [], "relationships": []}'

    class FakeModels:
        def generate_content(self, *, model, config, contents):
            captured["model"] = model
            captured["config"] = config
            return FakeResponse()

    class FakeGeminiClient:
        models = FakeModels()

    class FakeGenaiTypes:
        @staticmethod
        def GenerateContentConfig(**kwargs):
            return kwargs

    monkeypatch.setattr(graph, "genai_types", FakeGenaiTypes)
    monkeypatch.setattr(graph, "get_gemini_client", lambda: FakeGeminiClient())

    result = graph.extract_entity_graph_gemini("document text", "page.txt")

    assert result.document_id == "page.txt"
    assert captured["config"]["response_schema"] is graph.EntityGraph
    assert captured["config"]["system_instruction"] == graph.SYSTEM_PROMPT


def test_extract_entity_graph_dispatches_to_openai_when_provider_is_openai(monkeypatch) -> None:
    monkeypatch.setattr(graph, "get_provider_name", lambda: "openai")
    monkeypatch.setattr(graph, "extract_entity_graph_openai", lambda *a, **k: "openai-result")
    monkeypatch.setattr(graph, "extract_entity_graph_gemini", lambda *a, **k: "gemini-result")

    assert graph.extract_entity_graph("text", "page.txt") == "openai-result"


def test_extract_entity_graph_dispatches_to_gemini_when_provider_is_gemini(monkeypatch) -> None:
    monkeypatch.setattr(graph, "get_provider_name", lambda: "gemini")
    monkeypatch.setattr(graph, "extract_entity_graph_openai", lambda *a, **k: "openai-result")
    monkeypatch.setattr(graph, "extract_entity_graph_gemini", lambda *a, **k: "gemini-result")

    assert graph.extract_entity_graph("text", "page.txt") == "gemini-result"


def test_main_requires_input_file(capsys) -> None:
    assert graph.main([]) == 1

    captured = capsys.readouterr()
    assert "Usage: python entity_graph.py" in captured.err
