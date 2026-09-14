from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

import entity_graph as graph
import entity_registry as registry


def _make_extracted_entity(
    entity_id: str = "E1",
    entity_type: graph.EntityType = "person",
    canonical_name: str = "Jan Kowalski",
    mention_texts: list[str] | None = None,
    justification: str = "test justification",
) -> graph.ExtractedEntity:
    return graph.ExtractedEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        canonical_name=canonical_name,
        mention_texts=mention_texts if mention_texts is not None else [canonical_name],
        justification=justification,
    )


def _make_registry_entity(
    entity_id: str,
    entity_type: graph.EntityType,
    canonical_name: str,
    aliases: list[str] | None = None,
    source: Literal["llm", "lookup"] = "llm",
    mentions: list[registry.EntityMention] | None = None,
) -> registry.RegistryEntity:
    now = "2026-01-01T00:00:00+00:00"
    return registry.RegistryEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        canonical_name=canonical_name,
        aliases=aliases if aliases is not None else [canonical_name],
        source=source,
        created_at=now,
        updated_at=now,
        mentions=mentions if mentions is not None else [],
    )


def _mentions(count: int) -> list[registry.EntityMention]:
    mention = registry.EntityMention(
        document_id="d",
        local_entity_id="E1",
        mention_texts=["x"],
        justification="j",
        extracted_at="2026-01-01T00:00:00+00:00",
    )
    return [mention] * count


# --- normalize_name / next_entity_id ------------------------------------------


def test_normalize_name_strips_diacritics_and_case() -> None:
    assert registry.normalize_name("Lwów") == registry.normalize_name("Lwow")
    assert registry.normalize_name("JAN Kowalski") == registry.normalize_name("jan kowalski")
    assert registry.normalize_name("  Jan   Kowalski  ") == "jan kowalski"


def test_next_entity_id_starts_at_one_for_empty_registry() -> None:
    reg = registry.Registry()
    assert registry.next_entity_id(reg, "person") == "PER-0001"


def test_next_entity_id_recomputes_from_existing_max_ignoring_gaps() -> None:
    reg = registry.Registry(
        entities=[
            _make_registry_entity("PER-0001", "person", "Jan Kowalski"),
            _make_registry_entity("LOC-0009", "location", "Lwow"),
        ]
    )
    assert registry.next_entity_id(reg, "person") == "PER-0002"
    assert registry.next_entity_id(reg, "location") == "LOC-0010"


# --- resolve_via_lookup --------------------------------------------------------


def test_resolve_via_lookup_matches_alias_case_and_diacritic_insensitively() -> None:
    lookup = [
        registry.LookupEntity(
            entity_type="location", canonical_name="Lwow", aliases=["Leopolis", "Lviv", "Lwow"]
        )
    ]
    entity = _make_extracted_entity(
        entity_type="location", canonical_name="LEOPOLIS", mention_texts=["Leopolis"]
    )

    assert registry.resolve_via_lookup(entity, lookup) == "Lwow"


def test_resolve_via_lookup_returns_none_when_no_alias_matches() -> None:
    lookup = [registry.LookupEntity(entity_type="location", canonical_name="Lwow", aliases=["Leopolis"])]
    entity = _make_extracted_entity(entity_type="location", canonical_name="Krakow")

    assert registry.resolve_via_lookup(entity, lookup) is None


def test_resolve_via_lookup_ignores_matches_of_a_different_entity_type() -> None:
    lookup = [
        registry.LookupEntity(
            entity_type="person", canonical_name="Jan Kowalski", aliases=["Ioannes Kowalski"]
        )
    ]
    entity = _make_extracted_entity(entity_type="location", canonical_name="Ioannes Kowalski")

    assert registry.resolve_via_lookup(entity, lookup) is None


# --- find_registry_match --------------------------------------------------------


def test_find_registry_match_exact_normalized_canonical_name() -> None:
    existing = _make_registry_entity("PER-0001", "person", "Jan Kowalski")
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_type="person", canonical_name="jan kowalski", mention_texts=["jan kowalski"]
    )

    assert registry.find_registry_match(entity, reg, None) is existing


def test_find_registry_match_via_mention_text_overlap_with_existing_alias() -> None:
    existing = _make_registry_entity(
        "PER-0001", "person", "Jan Kowalski", aliases=["Jan Kowalski", "Ioannes Kowalski"]
    )
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_type="person", canonical_name="Ioannes Kowalski", mention_texts=["Ioannes Kowalski"]
    )

    assert registry.find_registry_match(entity, reg, None) is existing


def test_find_registry_match_fuzzy_merges_location_within_threshold() -> None:
    existing = _make_registry_entity("LOC-0001", "location", "Lwow")
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(entity_type="location", canonical_name="Lwof", mention_texts=["Lwof"])

    assert registry.find_registry_match(entity, reg, None) is existing


def test_find_registry_match_never_fuzzy_merges_person_type() -> None:
    existing = _make_registry_entity("PER-0001", "person", "Jan Kowalski")
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_type="person", canonical_name="Jan Kowalsk", mention_texts=["Jan Kowalsk"]
    )

    assert registry.find_registry_match(entity, reg, None) is None


def test_find_registry_match_returns_none_beyond_threshold() -> None:
    existing = _make_registry_entity("LOC-0001", "location", "Lwow")
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_type="location", canonical_name="Warszawa", mention_texts=["Warszawa"]
    )

    assert registry.find_registry_match(entity, reg, None) is None


def test_find_registry_match_logs_ambiguous_tie_to_review_queue() -> None:
    existing_a = _make_registry_entity("LOC-0001", "location", "Lwow")
    existing_b = _make_registry_entity("LOC-0002", "location", "Lwod")
    reg = registry.Registry(entities=[existing_a, existing_b])
    entity = _make_extracted_entity(entity_type="location", canonical_name="Lwoo", mention_texts=["Lwoo"])
    review_queue: list[dict] = []

    result = registry.find_registry_match(entity, reg, None, review_queue)

    assert result is None
    assert len(review_queue) == 1
    assert set(review_queue[0]["candidate_ids"]) == {"LOC-0001", "LOC-0002"}


# --- merge_entity / merge_document_graph ----------------------------------------


def test_merge_entity_creates_new_entry_when_no_match() -> None:
    reg = registry.Registry()
    entity = _make_extracted_entity(
        entity_id="E1", entity_type="location", canonical_name="Lwow", mention_texts=["Leopoliensis"]
    )
    review_queue: list[dict] = []

    result = registry.merge_entity(reg, entity, "doc-1.txt", [], review_queue)

    assert result.entity_id == "LOC-0001"
    assert result.canonical_name == "Lwow"
    assert set(result.aliases) == {"Lwow", "Leopoliensis"}
    assert result.source == "llm"
    assert len(result.mentions) == 1
    assert result.mentions[0].document_id == "doc-1.txt"
    assert reg.entities == [result]


def test_merge_entity_appends_mention_and_unions_aliases_on_match() -> None:
    existing = _make_registry_entity("LOC-0001", "location", "Lwow", aliases=["Lwow"])
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_id="E1", entity_type="location", canonical_name="Lwow", mention_texts=["Leopoliensis"]
    )

    result = registry.merge_entity(reg, entity, "doc-2.txt", [], [])

    assert result is existing
    assert "Leopoliensis" in result.aliases
    assert len(result.mentions) == 1
    assert result.mentions[0].document_id == "doc-2.txt"


def test_merge_entity_keeps_first_canonical_name_unless_lookup_forces_override() -> None:
    existing = _make_registry_entity("LOC-0001", "location", "Lwow", aliases=["Lwow", "Leopolis"])
    reg = registry.Registry(entities=[existing])
    entity = _make_extracted_entity(
        entity_id="E1", entity_type="location", canonical_name="Leopolis", mention_texts=["Leopolis"]
    )

    result = registry.merge_entity(reg, entity, "doc-3.txt", [], [])
    assert result.canonical_name == "Lwow"  # unchanged: first-writer-wins

    lookup = [registry.LookupEntity(entity_type="location", canonical_name="Lwów", aliases=["Lwow", "Leopolis"])]
    result2 = registry.merge_entity(reg, entity, "doc-4.txt", lookup, [])
    assert result2.canonical_name == "Lwów"  # lookup forces override


def test_merge_document_graph_returns_local_to_global_id_mapping() -> None:
    reg = registry.Registry()
    entity_graph = graph.EntityGraph(
        document_id="doc-1.txt",
        entities=[
            _make_extracted_entity(entity_id="E1", entity_type="person", canonical_name="Jan Kowalski"),
            _make_extracted_entity(entity_id="E2", entity_type="location", canonical_name="Lwow"),
        ],
    )

    mapping = registry.merge_document_graph(reg, entity_graph, "doc-1.txt", [], [])

    assert mapping == {"E1": "PER-0001", "E2": "LOC-0001"}
    assert len(reg.entities) == 2


# --- reconcile_with_lookup -------------------------------------------------------


def test_reconcile_with_lookup_merges_two_registry_entities_sharing_a_lookup_group() -> None:
    entity_a = _make_registry_entity("LOC-0001", "location", "Leopolis", aliases=["Leopolis"])
    entity_b = _make_registry_entity("LOC-0002", "location", "Lviv", aliases=["Lviv"])
    reg = registry.Registry(entities=[entity_a, entity_b])
    lookup = [
        registry.LookupEntity(entity_type="location", canonical_name="Lwow", aliases=["Leopolis", "Lviv", "Lwow"])
    ]

    merged_count = registry.reconcile_with_lookup(reg, lookup)

    assert merged_count == 1
    assert len(reg.entities) == 1
    survivor = reg.entities[0]
    assert survivor.entity_id == "LOC-0001"
    assert survivor.canonical_name == "Lwow"
    assert set(survivor.aliases) >= {"Leopolis", "Lviv"}
    assert survivor.merged_from == ["LOC-0002"]


def test_reconcile_with_lookup_is_a_noop_when_group_matches_at_most_one_entity() -> None:
    entity_a = _make_registry_entity("LOC-0001", "location", "Lwow", aliases=["Lwow"])
    reg = registry.Registry(entities=[entity_a])
    lookup = [
        registry.LookupEntity(entity_type="location", canonical_name="Lwow", aliases=["Leopolis", "Lviv", "Lwow"])
    ]

    merged_count = registry.reconcile_with_lookup(reg, lookup)

    assert merged_count == 0
    assert len(reg.entities) == 1


# --- load/save registry & lookup -------------------------------------------------


def test_load_registry_returns_empty_registry_when_file_missing(tmp_path: Path) -> None:
    result = registry.load_registry(tmp_path / "missing.json")

    assert result.entities == []
    assert result.processed_documents == []


def test_save_registry_round_trips_and_writes_atomically(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    reg = registry.Registry(entities=[_make_registry_entity("PER-0001", "person", "Jan Kowalski")])

    registry.save_registry(reg, path)

    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()

    reloaded = registry.load_registry(path)
    assert len(reloaded.entities) == 1
    assert reloaded.entities[0].entity_id == "PER-0001"


def test_load_lookup_returns_empty_list_when_file_missing(tmp_path: Path) -> None:
    assert registry.load_lookup(tmp_path / "missing.json") == []


def test_load_lookup_parses_seed_file(tmp_path: Path) -> None:
    path = tmp_path / "lookup.json"
    path.write_text(
        json.dumps([{"entity_type": "location", "canonical_name": "Lwow", "aliases": ["Leopolis"]}]),
        encoding="utf-8",
    )

    result = registry.load_lookup(path)

    assert len(result) == 1
    assert result[0].canonical_name == "Lwow"


# --- document_id_for / content_hash_for ------------------------------------------


def test_document_id_for_returns_posix_relative_path_from_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path
    nested = corpus_root / "23-2-52" / "0001.JPG.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("hello", encoding="utf-8")

    assert registry.document_id_for(nested, corpus_root) == "23-2-52/0001.JPG.txt"


def test_content_hash_for_is_stable_and_sha256_prefixed() -> None:
    first = registry.content_hash_for("some text")
    second = registry.content_hash_for("some text")

    assert first == second
    assert first.startswith("sha256:")
    assert registry.content_hash_for("different text") != first


# --- idempotency helpers ----------------------------------------------------------


def test_is_document_processed_true_when_hash_matches() -> None:
    reg = registry.Registry(
        processed_documents=[
            registry.ProcessedDocument(
                document_id="doc-1.txt",
                content_hash="sha256:abc",
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="doc-1.txt.entity_graph.json",
                entity_count=0,
                relationship_count=0,
            )
        ]
    )

    assert registry.is_document_processed(reg, "doc-1.txt", "sha256:abc") is True


def test_is_document_processed_false_when_hash_differs() -> None:
    reg = registry.Registry(
        processed_documents=[
            registry.ProcessedDocument(
                document_id="doc-1.txt",
                content_hash="sha256:abc",
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="doc-1.txt.entity_graph.json",
                entity_count=0,
                relationship_count=0,
            )
        ]
    )

    assert registry.is_document_processed(reg, "doc-1.txt", "sha256:different") is False


def test_remove_document_mentions_drops_llm_only_entities_but_keeps_lookup_sourced_ones() -> None:
    mention = registry.EntityMention(
        document_id="doc-1.txt",
        local_entity_id="E1",
        mention_texts=["Jan"],
        justification="j",
        extracted_at="2026-01-01T00:00:00+00:00",
    )
    llm_entity = _make_registry_entity("PER-0001", "person", "Jan Kowalski", mentions=[mention])
    lookup_entity = _make_registry_entity("LOC-0001", "location", "Lwow", source="lookup", mentions=[mention])
    reg = registry.Registry(
        entities=[llm_entity, lookup_entity],
        processed_documents=[
            registry.ProcessedDocument(
                document_id="doc-1.txt",
                content_hash="sha256:abc",
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="doc-1.txt.entity_graph.json",
                entity_count=2,
                relationship_count=0,
            )
        ],
    )

    registry.remove_document_mentions(reg, "doc-1.txt")

    remaining_ids = {e.entity_id for e in reg.entities}
    assert remaining_ids == {"LOC-0001"}
    assert reg.entities[0].mentions == []
    assert reg.processed_documents == []


# --- process_document -------------------------------------------------------------


def test_process_document_skips_and_does_not_call_extract_entity_graph_when_already_processed(
    tmp_path: Path, monkeypatch
) -> None:
    corpus_root = tmp_path
    doc_path = corpus_root / "page.txt"
    doc_path.write_text("some document text", encoding="utf-8")
    document_id = registry.document_id_for(doc_path, corpus_root)
    content_hash = registry.content_hash_for("some document text")

    reg = registry.Registry(
        processed_documents=[
            registry.ProcessedDocument(
                document_id=document_id,
                content_hash=content_hash,
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="page.txt.entity_graph.json",
                entity_count=0,
                relationship_count=0,
            )
        ]
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("extract_entity_graph should not be called for an already-processed document")

    monkeypatch.setattr(registry.graph, "extract_entity_graph", fail_if_called)

    result = registry.process_document(doc_path, corpus_root, reg, [], [])

    assert result is None


def test_process_document_calls_extract_entity_graph_and_merges_result(tmp_path: Path, monkeypatch) -> None:
    corpus_root = tmp_path
    doc_path = corpus_root / "page.txt"
    doc_path.write_text("some document text", encoding="utf-8")

    fake_graph = graph.EntityGraph(
        document_id="page.txt",
        entities=[_make_extracted_entity(entity_id="E1", entity_type="location", canonical_name="Lwow")],
    )
    monkeypatch.setattr(registry.graph, "extract_entity_graph", lambda *a, **k: fake_graph)

    reg = registry.Registry()
    output_path = registry.process_document(doc_path, corpus_root, reg, [], [])

    assert output_path is not None
    assert output_path == doc_path.with_suffix(doc_path.suffix + ".entity_graph.json")
    assert output_path.exists()
    assert len(reg.entities) == 1
    assert len(reg.processed_documents) == 1
    assert reg.processed_documents[0].document_id == "page.txt"


def test_process_document_skips_changed_document_without_reprocess_flag(tmp_path: Path, monkeypatch) -> None:
    corpus_root = tmp_path
    doc_path = corpus_root / "page.txt"
    doc_path.write_text("updated document text", encoding="utf-8")

    reg = registry.Registry(
        processed_documents=[
            registry.ProcessedDocument(
                document_id="page.txt",
                content_hash="sha256:stale",
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="page.txt.entity_graph.json",
                entity_count=0,
                relationship_count=0,
            )
        ]
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("extract_entity_graph should not be called without --reprocess")

    monkeypatch.setattr(registry.graph, "extract_entity_graph", fail_if_called)

    result = registry.process_document(doc_path, corpus_root, reg, [], [])

    assert result is None


def test_process_document_reprocess_removes_stale_mentions_before_remerging(tmp_path: Path, monkeypatch) -> None:
    corpus_root = tmp_path
    doc_path = corpus_root / "page.txt"
    doc_path.write_text("updated document text", encoding="utf-8")

    stale_mention = registry.EntityMention(
        document_id="page.txt",
        local_entity_id="E1",
        mention_texts=["Old Name"],
        justification="j",
        extracted_at="2026-01-01T00:00:00+00:00",
    )
    stale_entity = _make_registry_entity("PER-0001", "person", "Old Name", mentions=[stale_mention])
    reg = registry.Registry(
        entities=[stale_entity],
        processed_documents=[
            registry.ProcessedDocument(
                document_id="page.txt",
                content_hash="sha256:stale",
                processed_at="2026-01-01T00:00:00+00:00",
                entity_graph_path="page.txt.entity_graph.json",
                entity_count=1,
                relationship_count=0,
            )
        ],
    )

    fake_graph = graph.EntityGraph(
        document_id="page.txt",
        entities=[_make_extracted_entity(entity_id="E1", entity_type="person", canonical_name="New Name")],
    )
    monkeypatch.setattr(registry.graph, "extract_entity_graph", lambda *a, **k: fake_graph)

    result = registry.process_document(doc_path, corpus_root, reg, [], [], reprocess=True)

    assert result is not None
    person_entities = [e for e in reg.entities if e.entity_type == "person"]
    assert len(person_entities) == 1
    assert person_entities[0].canonical_name == "New Name"
    assert all("Old Name" not in m.mention_texts for m in person_entities[0].mentions)


# --- build_known_entities_context -------------------------------------------------


def test_build_known_entities_context_always_includes_lookup_and_caps_registry_entries() -> None:
    lookup = [registry.LookupEntity(entity_type="location", canonical_name="Lwow", aliases=["Leopolis", "Lviv"])]
    entities = [
        _make_registry_entity("PER-0001", "person", "Person One", mentions=_mentions(1)),
        _make_registry_entity("PER-0002", "person", "Person Two", mentions=_mentions(3)),
        _make_registry_entity("PER-0003", "person", "Person Three", mentions=_mentions(2)),
    ]
    reg = registry.Registry(entities=entities)

    context = registry.build_known_entities_context(reg, lookup, limit=2)

    assert "Lwow (aka Leopolis, Lviv)" in context
    assert context.index("Person Two") < context.index("Person Three")
    assert "Person One" not in context  # beyond the limit of 2
    assert "...and 1 more known entities not shown here." in context


def test_build_known_entities_context_excludes_registry_entities_already_covered_by_lookup() -> None:
    lookup = [registry.LookupEntity(entity_type="location", canonical_name="Lwow", aliases=["Leopolis"])]
    covered_entity = _make_registry_entity("LOC-0001", "location", "Lwow", mentions=_mentions(5))
    reg = registry.Registry(entities=[covered_entity])

    context = registry.build_known_entities_context(reg, lookup)

    assert context.count("Lwow") == 1  # only the lookup line, not duplicated via the registry entry


# --- process_directory (integration-style) ----------------------------------------


def test_process_directory_processes_matching_files_and_persists_registry(tmp_path: Path, monkeypatch) -> None:
    corpus_root = tmp_path
    batch_dir = corpus_root / "23-2-52"
    batch_dir.mkdir()
    (batch_dir / "0001.JPG.txt").write_text("first page text", encoding="utf-8")
    (batch_dir / "0002.JPG.txt").write_text("second page text", encoding="utf-8")

    call_count = {"n": 0}

    def fake_extract(document_text, document_id, known_entities_context=""):
        call_count["n"] += 1
        return graph.EntityGraph(
            document_id=document_id,
            entities=[_make_extracted_entity(entity_id="E1", entity_type="location", canonical_name="Lwow")],
        )

    monkeypatch.setattr(registry.graph, "extract_entity_graph", fake_extract)

    registry_path = corpus_root / "entity_registry" / "registry.json"
    lookup_path = corpus_root / "entity_registry" / "lookup.json"
    review_queue_path = corpus_root / "entity_registry" / "review_queue.json"

    registry.process_directory(
        batch_dir,
        registry_path=registry_path,
        lookup_path=lookup_path,
        review_queue_path=review_queue_path,
        corpus_root=corpus_root,
    )

    assert call_count["n"] == 2
    assert registry_path.exists()
    reloaded = registry.load_registry(registry_path)
    assert len(reloaded.entities) == 1  # both pages mention the same Lwow -> merged
    assert len(reloaded.processed_documents) == 2

    # Re-running should be a no-op: idempotent, no extra extraction calls.
    registry.process_directory(
        batch_dir,
        registry_path=registry_path,
        lookup_path=lookup_path,
        review_queue_path=review_queue_path,
        corpus_root=corpus_root,
    )
    assert call_count["n"] == 2


# --- CLI ----------------------------------------------------------------------------


def test_main_requires_directory_argument(capsys) -> None:
    with pytest.raises(SystemExit):
        registry.main([])

    captured = capsys.readouterr()
    assert captured.err  # argparse prints its own usage/error to stderr


def test_main_reports_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    exit_code = registry.main([str(missing)])

    assert exit_code == 1