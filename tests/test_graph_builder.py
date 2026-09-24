from __future__ import annotations

import networkx as nx
import pytest

import entity_graph as graph
import entity_registry as registry
import graph_builder as gb

NOW = "2026-01-01T00:00:00+00:00"
DOC3 = "23-2-52/0003.JPG.txt"
DOC4 = "23-2-52/0004.JPG.txt"
DOC5 = "23-2-52/0005.JPG.txt"


def _page(number: int, text: str = "text") -> gb.Page:
    return gb.Page(number, f"23-2-52/{number:04d}.JPG.txt", text)


def _entity(
    entity_id: str, name: str, mention_docs: list[str], local_id: str = "E1"
) -> registry.RegistryEntity:
    mentions = [
        registry.EntityMention(
            document_id=doc, local_entity_id=local_id, justification="t", extracted_at=NOW
        )
        for doc in mention_docs
    ]
    return registry.RegistryEntity(
        entity_id=entity_id,
        entity_type="person",
        canonical_name=name,
        source="llm",
        created_at=NOW,
        updated_at=NOW,
        mentions=mentions,
    )


def _span(page: int, quote: str | None = None) -> gb.EvidenceSpan:
    return gb.EvidenceSpan(page=page, quote=quote or f"quote from page {page}")


def _triple(subject: str, obj: str, pages: tuple[int, ...] = (5,)) -> gb.ExtractedTriple:
    return gb.ExtractedTriple(
        subject_id=subject,
        predicate="accusare",
        object_id=obj,
        attestation="inferred",
        evidence=[_span(p) for p in pages],
        reasoning="test",
    )


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def conn(tmp_path):
    connection = gb.open_db(tmp_path / "graph.sqlite")
    yield connection
    connection.close()


# --- pages, segments, windows ---------------------------------------------------


def test_find_pages_ignores_non_ocr_files_and_sorts_numerically(tmp_path):
    for name in ("0010.JPG.txt", "0002.JPG.txt", "0002.JPG.parsed.txt", "0002.JPG.words.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    pages = gb.find_pages(tmp_path, tmp_path)

    assert [p.number for p in pages] == [2, 10]
    assert pages[0].document_id == "0002.JPG.txt"


def test_split_segments_breaks_on_page_gaps():
    pages = [_page(n) for n in (1, 3, 4, 5, 6, 7)]

    segments = gb.split_segments(pages)

    assert [[p.number for p in seg] for seg in segments] == [[1], [3, 4, 5, 6, 7]]


def test_make_windows_overlaps_by_one_page(monkeypatch):
    monkeypatch.setattr(gb, "WINDOW_PAGES", 3)
    monkeypatch.setattr(gb, "WINDOW_OVERLAP", 1)
    segment = [_page(n) for n in (3, 4, 5, 6, 7)]

    windows = gb.make_windows(segment)

    assert [[p.number for p in w] for w in windows] == [[3, 4, 5], [5, 6, 7]]


def test_make_windows_single_page_is_one_window():
    assert len(gb.make_windows([_page(1)])) == 1


def test_make_windows_oversized_overlap_still_advances(monkeypatch):
    monkeypatch.setattr(gb, "WINDOW_PAGES", 3)
    monkeypatch.setattr(gb, "WINDOW_OVERLAP", 5)
    segment = [_page(n) for n in (3, 4, 5, 6, 7)]

    windows = gb.make_windows(segment)

    assert [[p.number for p in w] for w in windows] == [[3, 4, 5], [4, 5, 6], [5, 6, 7]]


# --- prompt building ------------------------------------------------------------


def test_entities_for_window_includes_mentions_and_carried_referents():
    on_page = _entity("PER-0001", "On Page", [DOC4])
    off_page = _entity("PER-0002", "Off Page", [DOC3])
    carried = _entity("PER-0003", "Carried", [DOC3])
    reg = registry.Registry(entities=[on_page, off_page, carried])
    carry_over = gb.CarryOverState(
        active_referents=[gb.ActiveReferent(entity_id="PER-0003", role="subject", last_page=3)]
    )

    offered = gb.entities_for_window(reg, [_page(4), _page(5)], carry_over)

    assert [e.entity_id for e in offered] == ["PER-0001", "PER-0003"]


def test_window_hash_changes_when_carry_over_changes():
    reg = registry.Registry(entities=[_entity("PER-0001", "A", [DOC4])])
    window = [_page(4)]
    empty = gb.CarryOverState()
    carried = gb.CarryOverState(open_clause="Idem coram")

    first = gb.window_hash(gb.build_window_prompt(reg, window, empty))
    again = gb.window_hash(gb.build_window_prompt(reg, window, empty))
    changed = gb.window_hash(gb.build_window_prompt(reg, window, carried))

    assert first == again
    assert first != changed
    assert first.startswith("sha256:")


# --- validation -----------------------------------------------------------------


def test_drop_unknown_ids_removes_invented_ids(capsys):
    extraction = gb.WindowExtraction(
        triples=[_triple("PER-0001", "PER-0002"), _triple("PER-0001", "PER-0999")],
        attributes=[
            gb.ExtractedAttribute(
                entity_id="PER-0999",
                key="occupation",
                value="furrier",
                attestation="normalized",
                evidence=[_span(4)],
                reasoning="test",
            )
        ],
        carry_over=gb.CarryOverState(
            active_referents=[
                gb.ActiveReferent(entity_id="PER-0001", role="subject", last_page=5),
                gb.ActiveReferent(entity_id="PER-0999", role="subject", last_page=5),
            ],
            open_clause="Idem coram",
        ),
    )

    cleaned = gb.drop_unknown_ids(extraction, {"PER-0001", "PER-0002"})

    assert [t.object_id for t in cleaned.triples] == ["PER-0002"]
    assert cleaned.attributes == []
    assert [r.entity_id for r in cleaned.carry_over.active_referents] == ["PER-0001"]
    assert cleaned.carry_over.open_clause == "Idem coram"
    assert "dropped 2" in capsys.readouterr().err


# --- saving ---------------------------------------------------------------------


def test_save_window_drops_evidence_for_pages_outside_window(conn):
    window = [_page(4), _page(5)]
    extraction = gb.WindowExtraction(
        triples=[
            _triple("PER-0001", "PER-0002", pages=(4, 9)),  # page 9 is not in the window
            _triple("PER-0001", "PER-0002", pages=(9,)),  # no valid evidence at all
        ]
    )

    gb.save_window(conn, "sha256:a", "23-2-52", window, extraction)

    assert _count(conn, "triples") == 1
    documents = [row[0] for row in conn.execute("SELECT document_id FROM evidence")]
    assert documents == [DOC4]


def test_save_window_replaces_stale_rows_for_same_pages(conn):
    window = [_page(4), _page(5)]
    first = gb.WindowExtraction(triples=[_triple("PER-0001", "PER-0002")])

    gb.save_window(conn, "sha256:a", "23-2-52", window, first)
    assert _count(conn, "triples") == 1 and _count(conn, "evidence") == 1

    gb.save_window(conn, "sha256:b", "23-2-52", window, gb.WindowExtraction())

    assert _count(conn, "windows") == 1
    assert _count(conn, "triples") == 0
    assert _count(conn, "evidence") == 0


# --- graph assembly -------------------------------------------------------------


def _two_person_registry() -> registry.Registry:
    return registry.Registry(
        entities=[
            _entity("PER-0002", "Simon Senior", [DOC4, DOC5], local_id="E2"),
            _entity("PER-0009", "Erasmus Jelonek", [DOC4], local_id="E3"),
        ]
    )


def test_nodes_carry_attributes_and_overlapping_windows_give_one_edge(conn):
    reg = _two_person_registry()
    occupation = gb.ExtractedAttribute(
        entity_id="PER-0009",
        key="occupation",
        value="furrier",
        attestation="normalized",
        evidence=[_span(4)],
        reasoning="pellionem",
    )
    # Page 5 sits in both windows, so both report the same fact with the same evidence.
    gb.save_window(
        conn,
        "sha256:a",
        "23-2-52",
        [_page(4), _page(5)],
        gb.WindowExtraction(triples=[_triple("PER-0002", "PER-0009")], attributes=[occupation]),
    )
    gb.save_window(
        conn,
        "sha256:b",
        "23-2-52",
        [_page(5), _page(6)],
        gb.WindowExtraction(triples=[_triple("PER-0002", "PER-0009")]),
    )
    g = nx.MultiDiGraph()

    gb.add_entity_nodes(g, reg, conn)
    gb.add_triple_edges(g, conn)

    assert g.nodes["PER-0009"]["attr_occupation"] == "furrier"
    assert g.nodes["PER-0002"]["documents"] == 2
    assert g.number_of_edges() == 1
    _, _, data = next(iter(g.edges(data=True)))
    assert data["source"] == "relations"
    assert data["predicate"] == "accusare"
    assert data["documents"] == DOC5


def test_add_triple_edges_skips_triples_pointing_at_unknown_nodes(conn):
    reg = registry.Registry(entities=[_entity("PER-0002", "Simon Senior", [DOC4])])
    gb.save_window(
        conn,
        "sha256:a",
        "23-2-52",
        [_page(4), _page(5)],
        gb.WindowExtraction(triples=[_triple("PER-0002", "PER-0999")]),
    )
    g = nx.MultiDiGraph()

    gb.add_entity_nodes(g, reg, conn)
    gb.add_triple_edges(g, conn)

    assert g.number_of_edges() == 0


def _write_page_graph(tmp_path, document_id: str, relationships) -> str:
    path = tmp_path / (document_id.replace("/", "_") + ".entity_graph.json")
    page_graph = graph.EntityGraph(document_id=document_id, relationships=relationships)
    path.write_text(page_graph.model_dump_json(), encoding="utf-8")
    return path.as_posix()


def _processed(document_id: str, path: str) -> registry.ProcessedDocument:
    return registry.ProcessedDocument(
        document_id=document_id,
        content_hash="sha256:x",
        processed_at=NOW,
        entity_graph_path=path,
        entity_count=2,
        relationship_count=1,
    )


def test_add_legacy_edges_translates_local_ids_to_registry_ids(tmp_path, conn):
    reg = _two_person_registry()
    path = _write_page_graph(
        tmp_path,
        DOC4,
        [
            graph.EntityRelationship(
                head_entity_id="E2", relation="accusavit", tail_entity_id="E3", evidence="ev"
            ),
            graph.EntityRelationship(  # E7 has no registry mention, so it is skipped
                head_entity_id="E2", relation="x", tail_entity_id="E7", evidence="ev"
            ),
        ],
    )
    reg.processed_documents.append(_processed(DOC4, path))
    g = nx.MultiDiGraph()
    gb.add_entity_nodes(g, reg, conn)

    gb.add_legacy_edges(g, reg)

    edges = list(g.edges(data=True))
    assert len(edges) == 1
    head, tail, data = edges[0]
    assert (head, tail) == ("PER-0002", "PER-0009")
    assert data["predicate"] == "accusavit"
    assert data["source"] == "entity_graph"
    assert data["attestation"] == "unspecified"


def test_add_legacy_edges_warns_about_missing_files(tmp_path, conn, capsys):
    reg = _two_person_registry()
    reg.processed_documents.append(_processed(DOC4, (tmp_path / "gone.json").as_posix()))
    g = nx.MultiDiGraph()
    gb.add_entity_nodes(g, reg, conn)

    gb.add_legacy_edges(g, reg)

    assert g.number_of_edges() == 0
    assert "missing entity graph file" in capsys.readouterr().err


# --- process_folder (Gemini mocked) ---------------------------------------------


def _fake_extract(calls: list[str]):
    def fake(prompt: str) -> gb.WindowExtraction:
        calls.append(prompt)
        return gb.WindowExtraction(
            triples=[_triple("PER-0002", "PER-0009", pages=(5,))],
            carry_over=gb.CarryOverState(
                active_referents=[
                    gb.ActiveReferent(entity_id="PER-0002", role="subject", last_page=5)
                ]
            ),
        )

    return fake


def test_process_folder_caches_windows_and_reruns_only_changed_ones(
    tmp_path, conn, monkeypatch, capsys
):
    monkeypatch.setattr(gb, "WINDOW_PAGES", 3)
    monkeypatch.setattr(gb, "WINDOW_OVERLAP", 1)
    folder = tmp_path / "23-2-52"
    folder.mkdir()
    for number in range(3, 8):
        (folder / f"{number:04d}.JPG.txt").write_text(f"text {number}", encoding="utf-8")
    reg = registry.Registry(
        entities=[
            _entity("PER-0002", "Simon Senior", [DOC4, DOC5]),
            _entity("PER-0009", "Erasmus Jelonek", [DOC4, DOC5]),
        ]
    )
    calls: list[str] = []
    monkeypatch.setattr(gb, "extract_window", _fake_extract(calls))

    gb.process_folder(folder, tmp_path, reg, conn)
    assert len(calls) == 2  # windows [3,4,5] and [5,6,7]
    assert _count(conn, "windows") == 2

    capsys.readouterr()
    gb.process_folder(folder, tmp_path, reg, conn)
    assert len(calls) == 2  # everything cached
    assert capsys.readouterr().out.count("skip (cached)") == 2

    (folder / "0006.JPG.txt").write_text("corrected text 6", encoding="utf-8")
    gb.process_folder(folder, tmp_path, reg, conn)
    assert len(calls) == 3  # only [5,6,7] re-ran
    assert _count(conn, "windows") == 2


def test_process_folder_resets_carry_over_across_page_gaps(tmp_path, conn, monkeypatch):
    folder = tmp_path / "23-2-52"
    folder.mkdir()
    for number in (1, 3):  # page 2 is missing
        (folder / f"{number:04d}.JPG.txt").write_text(f"text {number}", encoding="utf-8")
    # Known on page 1, so it survives drop_unknown_ids and IS carried out of window 1.
    reg = registry.Registry(
        entities=[
            _entity("PER-0002", "Simon Senior", ["23-2-52/0001.JPG.txt", DOC3]),
            _entity("PER-0009", "Erasmus Jelonek", [DOC3]),
        ]
    )
    calls: list[str] = []
    monkeypatch.setattr(gb, "extract_window", _fake_extract(calls))

    gb.process_folder(folder, tmp_path, reg, conn)

    assert len(calls) == 2
    assert '"active_referents": []' in calls[1]  # nothing carried across the gap


# --- command line ---------------------------------------------------------------


def test_parse_args_requires_directory_unless_build_only():
    with pytest.raises(SystemExit):
        gb.parse_args([])

    assert gb.parse_args(["--build-only"]).directory is None
