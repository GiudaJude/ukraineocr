from pathlib import Path

import pytest

import openai_entity_graph as graph


def test_get_client_reads_api_key_from_environment(monkeypatch) -> None:
    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(graph, "OpenAI", FakeOpenAI)

    client = graph.get_client()

    assert client.api_key == "test-key"


def test_default_output_path_appends_entity_graph_suffix(tmp_path: Path) -> None:
    input_path = tmp_path / "page.txt"
    input_path.write_text("hello", encoding="utf-8")

    assert graph.default_output_path(input_path) == tmp_path / "page.txt.entity_graph.json"


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


def test_main_requires_input_file(capsys) -> None:
    assert graph.main([]) == 1

    captured = capsys.readouterr()
    assert "Usage: python openai_entity_graph.py" in captured.err
