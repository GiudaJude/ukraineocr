from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys

import networkx as nx

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import entity_graph as graph
import entity_registry as registry

# TODO: add OpenAI provider support (mirror entity_graph.extract_entity_graph_openai).
# Gemini is the only provider wired up for now.
GEMINI_MODEL_RELATIONS = os.getenv("GEMINI_MODEL_RELATIONS", graph.GEMINI_MODEL_NER)
GEMINI_RELATIONS_MAX_OUTPUT_TOKENS = int(
    os.getenv("GEMINI_RELATIONS_MAX_OUTPUT_TOKENS", "65536")
)
WINDOW_PAGES = int(os.getenv("GRAPH_WINDOW_PAGES", "3"))
WINDOW_OVERLAP = int(os.getenv("GRAPH_WINDOW_OVERLAP", "1"))
DEFAULT_DB_PATH = Path("entity_registry/graph.sqlite")

Attestation = Literal["attested", "normalized", "inferred"]


class EvidenceSpan(BaseModel):
    page: int
    quote: str  # verbatim from the text; one span per sentence/clause used


class ExtractedTriple(BaseModel):
    subject_id: str  # registry ID, e.g. "PER-0002"
    predicate: str  # lemma, e.g. "esse", "actor_against"
    object_id: str  # registry ID, e.g. "ORG-0003"
    attestation: Attestation
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reasoning: str  # how the spans connect, e.g. "unnamed subject of 'esse' is Simon"


class ExtractedAttribute(BaseModel):
    entity_id: str
    key: str  # e.g. "occupation", "origin", "role"
    value: str  # e.g. "furrier", "Cracow", "councillor"
    attestation: Attestation
    evidence: list[EvidenceSpan] = Field(min_length=1)
    reasoning: str


class ActiveReferent(BaseModel):
    entity_id: str  # registry ID still "in scope" at the end of the window
    role: str  # e.g. "grammatical subject", "last named person"
    last_page: int


class CarryOverState(BaseModel):
    active_referents: list[ActiveReferent] = Field(default_factory=list)
    open_clause: str = ""  # unfinished sentence tail, verbatim, or "" if none


class WindowExtraction(BaseModel):
    triples: list[ExtractedTriple] = Field(default_factory=list)
    attributes: list[ExtractedAttribute] = Field(default_factory=list)
    carry_over: CarryOverState = Field(default_factory=CarryOverState)


SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    hash TEXT PRIMARY KEY,
    folder TEXT NOT NULL,
    pages TEXT NOT NULL,          -- JSON list of document_ids in the window
    processed_at TEXT NOT NULL,
    carry_state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_hash TEXT NOT NULL REFERENCES windows(hash) ON DELETE CASCADE,
    subject_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL,
    attestation TEXT NOT NULL,
    reasoning TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attributes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    window_hash TEXT NOT NULL REFERENCES windows(hash) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    attestation TEXT NOT NULL,
    reasoning TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    triple_id INTEGER REFERENCES triples(id) ON DELETE CASCADE,
    attribute_id INTEGER REFERENCES attributes(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    quote TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_attributes_entity ON attributes(entity_id);
"""


def open_db(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


PAGE_FILE_RE = re.compile(r"^(\d+)\.JPG\.txt$", re.IGNORECASE)


@dataclass(frozen=True)
class Page:
    number: int
    document_id: str  # same ID the registry uses, e.g. "23-2-52/0004.JPG.txt"
    text: str


def find_pages(folder: Path, corpus_root: Path) -> list[Page]:
    pages = []
    for path in folder.iterdir():
        match = PAGE_FILE_RE.match(path.name)
        if match:
            document_id = registry.document_id_for(path, corpus_root)
            pages.append(Page(int(match.group(1)), document_id, path.read_text(encoding="utf-8")))
    return sorted(pages, key=lambda page: page.number)


def split_segments(pages: list[Page]) -> list[list[Page]]:
    segments: list[list[Page]] = []
    for page in pages:
        if segments and page.number == segments[-1][-1].number + 1:
            segments[-1].append(page)
        else:
            segments.append([page])
    return segments


def make_windows(segment: list[Page]) -> list[list[Page]]:
    step = max(WINDOW_PAGES - WINDOW_OVERLAP, 1)
    windows = []
    for start in range(0, len(segment), step):
        windows.append(segment[start : start + WINDOW_PAGES])
        if start + WINDOW_PAGES >= len(segment):
            break
    return windows


RELATIONS_SYSTEM_PROMPT = """ROLE: You are an expert paleographer and Latin/Old Polish
philologist working on 16th-17th century Lviv council records.

TASK: Extract relationship triples and entity attributes from a window of
consecutive OCR pages. Return only schema-compliant JSON, with no commentary.
"""

RELATIONS_USER_PROMPT_TEMPLATE = """Extract relationships from the pages below.

Known entities (the ONLY entity IDs you may use):
{entities_context}

State carried over from the previous window (may be empty):
{carry_over_json}

Rules:
- Every subject_id, object_id and entity_id must be one of the known entity IDs.
- predicate is the lemma of the governing verb (e.g. esse, agere, solvere) or a
  short snake_case relation (e.g. member_of, actor_against). More is better.
- Attributes are facts about one entity: occupation, origin, role, title.
  E.g. "Pellionem" after Erasmus Jelonek -> occupation: furrier.
- Latin drops subjects. If a subject is unnamed, resolve it using the carried-over
  state and earlier text. A subject and its predicate or occupation may be several
  sentences or a page apart; link them and say how in `reasoning`.
- attestation: "attested" = stated outright; "normalized" = stated in a changed
  form (case ending, spelling); "inferred" = deduced from grammar or context,
  including any subject or link resolved across sentences or pages.
- Do not silently correct uncertain or OCR-corrupted names. If you are unsure which
  entity is meant, omit the fact.
- evidence: one span per sentence or clause used, quoted verbatim from the text,
  with `page` set to the number in that page's marker.
- carry_over: list the entities still in scope at the end of the last page and
  any unfinished sentence tail (verbatim).

Pages:
{window_text}
"""


def entities_for_window(
    reg: registry.Registry, window: list[Page], carry_over: CarryOverState
) -> list[registry.RegistryEntity]:
    document_ids = {page.document_id for page in window}
    carried_ids = {referent.entity_id for referent in carry_over.active_referents}
    return [
        entity
        for entity in reg.entities
        if entity.entity_id in carried_ids
        or any(mention.document_id in document_ids for mention in entity.mentions)
    ]


def format_entities(entities: list[registry.RegistryEntity]) -> str:
    lines = []
    for entity in entities:
        aka = ", ".join(a for a in entity.aliases if a != entity.canonical_name)
        line = f"{entity.entity_id} | {entity.entity_type} | {entity.canonical_name}"
        lines.append(f"{line} | aka {aka}" if aka else line)
    return "\n".join(lines)


def format_window_text(window: list[Page]) -> str:
    return "\n\n".join(f"=== PAGE {page.number} ===\n{page.text}" for page in window)


def build_window_prompt(
    reg: registry.Registry, window: list[Page], carry_over: CarryOverState
) -> str:
    return RELATIONS_USER_PROMPT_TEMPLATE.format(
        entities_context=format_entities(entities_for_window(reg, window, carry_over)),
        carry_over_json=carry_over.model_dump_json(indent=2),
        window_text=format_window_text(window),
    )


def window_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def extract_window(prompt: str) -> WindowExtraction:
    if graph.genai_types is None:
        raise RuntimeError("The `google-genai` package is required for GEMINI_API_KEY.")

    config = graph.genai_types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=GEMINI_RELATIONS_MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=WindowExtraction,
        system_instruction=RELATIONS_SYSTEM_PROMPT,
    )
    response = graph.get_gemini_client().models.generate_content(
        model=GEMINI_MODEL_RELATIONS,
        config=config,
        contents=[{"text": prompt}],
    )
    raw_output = getattr(response, "text", "") or ""
    if not raw_output:
        raise RuntimeError("Gemini returned an empty response for this window.")
    return WindowExtraction.model_validate_json(raw_output)


def drop_unknown_ids(extraction: WindowExtraction, known_ids: set[str]) -> WindowExtraction:
    triples = [
        t for t in extraction.triples if t.subject_id in known_ids and t.object_id in known_ids
    ]
    attributes = [a for a in extraction.attributes if a.entity_id in known_ids]
    referents = [r for r in extraction.carry_over.active_referents if r.entity_id in known_ids]

    dropped = (len(extraction.triples) - len(triples)) + (
        len(extraction.attributes) - len(attributes)
    )
    if dropped:
        print(f"warning: dropped {dropped} fact(s) with unknown entity IDs", file=sys.stderr)

    carry_over = CarryOverState(
        active_referents=referents, open_clause=extraction.carry_over.open_clause
    )
    return WindowExtraction(triples=triples, attributes=attributes, carry_over=carry_over)


def resolve_spans(spans: list[EvidenceSpan], doc_by_page: dict[int, str]) -> list[tuple[str, str]]:
    return [(doc_by_page[span.page], span.quote) for span in spans if span.page in doc_by_page]


def insert_evidence(
    conn: sqlite3.Connection,
    resolved: list[tuple[str, str]],
    *,
    triple_id: int | None = None,
    attribute_id: int | None = None,
) -> None:
    for document_id, quote in resolved:
        conn.execute(
            "INSERT INTO evidence (triple_id, attribute_id, document_id, quote) VALUES (?, ?, ?, ?)",
            (triple_id, attribute_id, document_id, quote),
        )


def insert_triple(
    conn: sqlite3.Connection,
    window_key: str,
    triple: ExtractedTriple,
    doc_by_page: dict[int, str],
) -> None:
    resolved = resolve_spans(triple.evidence, doc_by_page)
    if not resolved:
        return  # never store a claim with no valid evidence

    cursor = conn.execute(
        "INSERT INTO triples "
        "(window_hash, subject_id, predicate, object_id, attestation, reasoning) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            window_key,
            triple.subject_id,
            triple.predicate,
            triple.object_id,
            triple.attestation,
            triple.reasoning,
        ),
    )
    insert_evidence(conn, resolved, triple_id=cursor.lastrowid)


def insert_attribute(
    conn: sqlite3.Connection,
    window_key: str,
    attribute: ExtractedAttribute,
    doc_by_page: dict[int, str],
) -> None:
    resolved = resolve_spans(attribute.evidence, doc_by_page)
    if not resolved:
        return  # never store a fact with no valid evidence

    cursor = conn.execute(
        "INSERT INTO attributes "
        "(window_hash, entity_id, key, value, attestation, reasoning) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            window_key,
            attribute.entity_id,
            attribute.key,
            attribute.value,
            attribute.attestation,
            attribute.reasoning,
        ),
    )
    insert_evidence(conn, resolved, attribute_id=cursor.lastrowid)


def save_window(
    conn: sqlite3.Connection,
    window_key: str,
    folder: str,
    window: list[Page],
    extraction: WindowExtraction,
) -> None:
    doc_by_page = {page.number: page.document_id for page in window}
    page_ids = json.dumps([page.document_id for page in window])

    with conn:
        conn.execute(
            "DELETE FROM windows WHERE hash = ? OR (folder = ? AND pages = ?)",
            (window_key, folder, page_ids),
        )
        conn.execute(
            "INSERT INTO windows (hash, folder, pages, processed_at, carry_state_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                window_key,
                folder,
                page_ids,
                datetime.now(timezone.utc).isoformat(),
                extraction.carry_over.model_dump_json(),
            ),
        )
        for triple in extraction.triples:
            insert_triple(conn, window_key, triple, doc_by_page)
        for attribute in extraction.attributes:
            insert_attribute(conn, window_key, attribute, doc_by_page)


def process_folder(
    folder: Path, corpus_root: Path, reg: registry.Registry, conn: sqlite3.Connection
) -> None:
    folder_id = folder.resolve().relative_to(corpus_root.resolve()).as_posix()
    for segment in split_segments(find_pages(folder, corpus_root)):
        carry_over = CarryOverState()  # a page gap resets the carried-over state
        for window in make_windows(segment):
            label = f"{folder_id} pages {window[0].number}-{window[-1].number}"
            prompt = build_window_prompt(reg, window, carry_over)
            key = window_hash(prompt)

            cached = conn.execute(
                "SELECT carry_state_json FROM windows WHERE hash = ?", (key,)
            ).fetchone()
            if cached:
                print(f"skip (cached): {label}")
                carry_over = CarryOverState.model_validate_json(cached[0])
                continue

            known_ids = {e.entity_id for e in entities_for_window(reg, window, carry_over)}
            extraction = drop_unknown_ids(extract_window(prompt), known_ids)
            save_window(conn, key, folder_id, window, extraction)
            carry_over = extraction.carry_over
            print(f"extracted: {label} ({len(extraction.triples)} triples)")


def add_entity_nodes(
    g: nx.MultiDiGraph, reg: registry.Registry, conn: sqlite3.Connection
) -> None:
    attrs: dict[str, dict[str, list[str]]] = {}
    for entity_id, key, value in conn.execute(
        "SELECT entity_id, key, value FROM attributes ORDER BY id"
    ):
        values = attrs.setdefault(entity_id, {}).setdefault(key, [])
        if value not in values:
            values.append(value)

    for entity in reg.entities:
        extra = {
            f"attr_{key}": "; ".join(values)
            for key, values in attrs.get(entity.entity_id, {}).items()
        }
        g.add_node(
            entity.entity_id,
            entity_type=entity.entity_type,
            canonical_name=entity.canonical_name,
            aliases="; ".join(entity.aliases),
            documents=len({mention.document_id for mention in entity.mentions}),
            **extra,
        )


def add_triple_edges(g: nx.MultiDiGraph, conn: sqlite3.Connection) -> None:
    evidence: dict[int, list[tuple[str, str]]] = {}
    for triple_id, document_id, quote in conn.execute(
        "SELECT triple_id, document_id, quote FROM evidence "
        "WHERE triple_id IS NOT NULL ORDER BY id"
    ):
        evidence.setdefault(triple_id, []).append((document_id, quote))

    seen: set[tuple] = set()
    for row in conn.execute(
        "SELECT id, subject_id, predicate, object_id, attestation, reasoning "
        "FROM triples ORDER BY id"
    ):
        triple_id, subject_id, predicate, object_id, attestation, reasoning = row
        spans = evidence.get(triple_id, [])
        signature = (subject_id, predicate, object_id, frozenset(spans))
        if signature in seen or subject_id not in g or object_id not in g:
            continue
        seen.add(signature)
        g.add_edge(
            subject_id,
            object_id,
            key=triple_id,
            predicate=predicate,
            attestation=attestation,
            reasoning=reasoning,
            documents="; ".join(sorted({document_id for document_id, _ in spans})),
            evidence=" | ".join(quote for _, quote in spans),
            source="relations",
        )


def add_legacy_edges(g: nx.MultiDiGraph, reg: registry.Registry) -> None:
    local_to_global = {
        (mention.document_id, mention.local_entity_id): entity.entity_id
        for entity in reg.entities
        for mention in entity.mentions
    }
    for doc in reg.processed_documents:
        path = Path(doc.entity_graph_path)
        if not path.exists():
            print(f"warning: missing entity graph file {path}", file=sys.stderr)
            continue
        page_graph = graph.EntityGraph.model_validate_json(path.read_text(encoding="utf-8"))
        for index, rel in enumerate(page_graph.relationships):
            head = local_to_global.get((doc.document_id, rel.head_entity_id))
            tail = local_to_global.get((doc.document_id, rel.tail_entity_id))
            if head is None or tail is None:
                continue
            g.add_edge(
                head,
                tail,
                key=f"page:{doc.document_id}:{index}",
                predicate=rel.relation,
                attestation="unspecified",
                reasoning="",
                documents=doc.document_id,
                evidence=rel.evidence,
                source="entity_graph",
            )


def build_graph(reg: registry.Registry, conn: sqlite3.Connection) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    add_entity_nodes(g, reg, conn)
    add_triple_edges(g, conn)
    add_legacy_edges(g, reg)
    return g


def export_graph(g: nx.MultiDiGraph, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    graphml_path = output_stem.with_name(output_stem.name + ".graphml")
    json_path = output_stem.with_name(output_stem.name + ".json")

    nx.write_graphml(g, graphml_path)
    data = nx.node_link_data(g, edges="edges")
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return graphml_path, json_path


def print_summary(g: nx.MultiDiGraph, top: int = 5) -> None:
    print(f"nodes: {g.number_of_nodes()}, edges: {g.number_of_edges()}")
    by_source: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        by_source[data["source"]] = by_source.get(data["source"], 0) + 1
    print("edges by source: " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())))

    print(f"top {top} most connected entities:")
    for node, degree in sorted(g.degree, key=lambda item: item[1], reverse=True)[:top]:
        print(f"  {node} {g.nodes[node]['canonical_name']}: {degree} edges")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cross-page relations with Gemini and build a NetworkX entity graph."
    )
    parser.add_argument(
        "directory", type=Path, nargs="?", help="Directory of *.JPG.txt pages to extract from"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Skip Gemini extraction; build the graph from what is already in the database",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("entity_registry/graph"),
        help="Output path without extension (writes .graphml and .json)",
    )
    args = parser.parse_args(argv)
    if args.directory is None and not args.build_only:
        parser.error("directory is required unless --build-only is given")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.directory is not None and not args.directory.exists():
        print(f"Directory not found: {args.directory}", file=sys.stderr)
        return 1

    reg = registry.load_registry()
    if not reg.entities:
        print("Registry is empty. Run entity_registry.py first.", file=sys.stderr)
        return 1

    conn = open_db()
    try:
        if args.directory is not None and not args.build_only:
            process_folder(args.directory, Path.cwd(), reg, conn)
        g = build_graph(reg, conn)
    finally:
        conn.close()

    graphml_path, json_path = export_graph(g, args.output)
    print_summary(g)
    print(f"wrote {graphml_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
