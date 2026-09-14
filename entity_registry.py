"""Corpus-wide entity registry.

Incrementally merges per-document entity extractions (produced by
entity_graph.extract_entity_graph) into one persistaent registry, so that
name/place varients referring to the same real-world entity. Example: a
Latinized name and its vernacular form, or a place's historical Latin/
Polish/Ukrainian names, accumulate into a single canonical record with
full provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import entity_graph as graph

from post_tokenization import levenshtein

DEFAULT_REGISTRY_DIR = Path("entity_registry")
DEFAULT_REVIEW_QUEUE_PATH = DEFAULT_REGISTRY_DIR / "review_queue.json"
DEFAULT_REGISTRY_PATH = DEFAULT_REGISTRY_DIR / "registry.json"
DEFAULT_LOOKUP_PATH = DEFAULT_REGISTRY_DIR / "lookup.json"
DEFAULT_REVIEW_QUEUE_PATH = DEFAULT_REGISTRY_DIR / "review_queue.json"

DOCUMENT_GLOB = "*.JPG.txt"

# Fuzzy (edit-distance) auto-merging only applies to these types. A false
# merge of two different people is worse than a missed merge, so "person"
# (and "date"/"currency", where distance is meaningless) are excluded.
FUZZY_ENTITY_TYPES = {"location", "organization"}
FUZZY_MAX_DISTANCE_RATIO = float(os.getenv("ENTITY_FUZZY_MAX_DISTANCE_RATIO", "0.15"))
FUZZY_MIN_LENGTH = 4

KNOWN_ENTITIES_CONTEXT_LIMIT = int(os.getenv("ENTITY_CONTEXT_LIMIT", "200"))

_ID_PREFIXES: dict[str, str] = {
    "person": "PER",
    "organization": "ORG",
    "location": "LOC",
    "date": "DATE",
    "currency": "CUR",
}


class EntityMention(BaseModel):
    document_id: str
    local_entity_id: str
    mention_texts: list[str] = Field(default_factory=list)
    justification: str
    extracted_at: str


class RegistryEntity(BaseModel):
    entity_id: str
    entity_type: graph.EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    source: Literal["llm", "lookup"]
    merged_from: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    mentions: list[EntityMention] = Field(default_factory=list)


class ProcessedDocument(BaseModel):
    document_id: str
    content_hash: str
    processed_at: str
    entity_graph_path: str
    entity_count: int
    relationship_count: int


class Registry(BaseModel):
    schema_version: int = 1
    updated_at: str = ""
    entities: list[RegistryEntity] = Field(default_factory=list)
    processed_documents: list[ProcessedDocument] = Field(default_factory=list)


class LookupEntity(BaseModel):
    entity_type: graph.EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    notes: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(name: str) -> str:
    """Fold a name to a diacritic- and case-insensitive comparison key.

    NFKD decomposition splits an accented letter into a base letter plus a
    combining mark (e.g. "o" + U+0301 for "ó"); dropping every combining
    mark then leaves just the plain base letters. This lets "Lwów" and
    "Lwow" compare equal without ever touching the original spelling
    stored anywhere in the registry.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def _dedup_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        key = normalize_name(name)
        if key and key not in seen:
            seen.add(key)
            result.append(name)
    return result


# --- Persistence -------------------------------------------------------------


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> Registry:
    if not path.exists():
        return Registry()
    return Registry.model_validate_json(path.read_text(encoding="utf-8"))


def save_registry(registry: Registry, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    registry.updated_at = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(registry.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)  # atomic rename: never leaves a half-written registry.json


def load_lookup(path: Path = DEFAULT_LOOKUP_PATH) -> list[LookupEntity]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LookupEntity.model_validate(item) for item in raw]


def document_id_for(path: Path, corpus_root: Path) -> str:
    return path.resolve().relative_to(corpus_root.resolve()).as_posix()


def content_hash_for(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def next_entity_id(registry: Registry, entity_type: str) -> str:
    prefix = _ID_PREFIXES[entity_type]
    max_seen = 0
    for entity in registry.entities:
        if entity.entity_id.startswith(prefix + "-"):
            try:
                max_seen = max(max_seen, int(entity.entity_id.split("-", 1)[1]))
            except ValueError:
                continue
    return f"{prefix}-{max_seen + 1:04d}"


# --- Matching ------------------------------------------------------------------


def resolve_via_lookup(entity: graph.ExtractedEntity, lookup: list[LookupEntity]) -> str | None:
    candidates = {normalize_name(entity.canonical_name)}
    candidates.update(normalize_name(m) for m in entity.mention_texts)

    for group in lookup:
        if group.entity_type != entity.entity_type:
            continue
        group_aliases = {normalize_name(a) for a in group.aliases}
        group_aliases.add(normalize_name(group.canonical_name))
        if candidates & group_aliases:
            return group.canonical_name
    return None


def find_registry_match(
    entity: graph.ExtractedEntity,
    registry: Registry,
    lookup_canonical: str | None,
    review_queue: list[dict] | None = None,
) -> RegistryEntity | None:
    candidates = {normalize_name(lookup_canonical or entity.canonical_name)}
    candidates.update(normalize_name(m) for m in entity.mention_texts)

    same_type = [e for e in registry.entities if e.entity_type == entity.entity_type]

    for existing in same_type:
        existing_names = {normalize_name(existing.canonical_name)}
        existing_names.update(normalize_name(a) for a in existing.aliases)
        if candidates & existing_names:
            return existing

    if entity.entity_type not in FUZZY_ENTITY_TYPES:
        return None

    target = normalize_name(lookup_canonical or entity.canonical_name)
    if len(target) < FUZZY_MIN_LENGTH:
        return None

    threshold = max(1, round(len(target) * FUZZY_MAX_DISTANCE_RATIO))
    scored = [
        (levenshtein(target, normalize_name(existing.canonical_name)), existing)
        for existing in same_type
    ]
    within_threshold = sorted(((d, e) for d, e in scored if d <= threshold), key=lambda pair: pair[0])
    if not within_threshold:
        return None

    best_distance = within_threshold[0][0]
    best_matches = [e for d, e in within_threshold if d == best_distance]
    if len(best_matches) > 1:
        if review_queue is not None:
            review_queue.append(
                {
                    "new_name": entity.canonical_name,
                    "candidate_ids": [e.entity_id for e in best_matches],
                    "distance": best_distance,
                }
            )
        return None
    return best_matches[0]


def merge_entity(
    registry: Registry,
    entity: graph.ExtractedEntity,
    document_id: str,
    lookup: list[LookupEntity],
    review_queue: list[dict],
) -> RegistryEntity:
    now = _now()
    lookup_canonical = resolve_via_lookup(entity, lookup)
    match = find_registry_match(entity, registry, lookup_canonical, review_queue)

    mention = EntityMention(
        document_id=document_id,
        local_entity_id=entity.entity_id,
        mention_texts=list(entity.mention_texts) or [entity.canonical_name],
        justification=entity.justification,
        extracted_at=now,
    )

    if match is None:
        canonical_name = lookup_canonical or entity.canonical_name
        new_entity = RegistryEntity(
            entity_id=next_entity_id(registry, entity.entity_type),
            entity_type=entity.entity_type,
            canonical_name=canonical_name,
            aliases=_dedup_names([canonical_name, *entity.mention_texts]),
            source="lookup" if lookup_canonical else "llm",
            created_at=now,
            updated_at=now,
            mentions=[mention],
        )
        registry.entities.append(new_entity)
        return new_entity

    match.mentions.append(mention)
    match.aliases = _dedup_names([*match.aliases, entity.canonical_name, *entity.mention_texts])
    if lookup_canonical:
        match.canonical_name = lookup_canonical
    match.updated_at = now
    return match


def merge_document_graph(
    registry: Registry,
    entity_graph: graph.EntityGraph,
    document_id: str,
    lookup: list[LookupEntity],
    review_queue: list[dict],
) -> dict[str, str]:
    """Merge every entity in entity_graph into registry.

    Returns the local -> global entity_id mapping for this document (handy
    later if relationship-graph merging is ever added).
    """
    local_to_global: dict[str, str] = {}
    for entity in entity_graph.entities:
        merged = merge_entity(registry, entity, document_id, lookup, review_queue)
        local_to_global[entity.entity_id] = merged.entity_id
    return local_to_global


def reconcile_with_lookup(registry: Registry, lookup: list[LookupEntity]) -> int:
    """Merge registry entities that a lookup.json group now links together.

    Lets you add a lookup entry after the fact and have it retroactively
    collapse entities that were registered separately before the entry
    existed, without re-running any LLM extraction.
    """
    merges = 0
    for group in lookup:
        group_aliases = {normalize_name(a) for a in group.aliases}
        group_aliases.add(normalize_name(group.canonical_name))

        matches = [
            e
            for e in registry.entities
            if e.entity_type == group.entity_type
            and (
                {normalize_name(e.canonical_name), *(normalize_name(a) for a in e.aliases)}
                & group_aliases
            )
        ]
        if len(matches) < 2:
            continue

        matches.sort(key=lambda e: e.entity_id)
        survivor, *losers = matches
        for loser in losers:
            survivor.mentions.extend(loser.mentions)
            survivor.aliases = _dedup_names([*survivor.aliases, *loser.aliases])
            survivor.merged_from = _dedup_names(
                [*survivor.merged_from, loser.entity_id, *loser.merged_from]
            )
            registry.entities.remove(loser)
            merges += 1
        survivor.canonical_name = group.canonical_name
        survivor.updated_at = _now()

    return merges


def build_known_entities_context(
    registry: Registry,
    lookup: list[LookupEntity],
    limit: int = KNOWN_ENTITIES_CONTEXT_LIMIT,
) -> str:
    lines: list[str] = []

    lookup_covered: set[str] = set()
    for group in lookup:
        lookup_covered.add(normalize_name(group.canonical_name))
        aka = ", ".join(
            a for a in group.aliases if normalize_name(a) != normalize_name(group.canonical_name)
        )
        lines.append(f"{group.canonical_name} (aka {aka})" if aka else group.canonical_name)

    remaining = [
        e for e in registry.entities if normalize_name(e.canonical_name) not in lookup_covered
    ]
    remaining.sort(key=lambda e: len(e.mentions), reverse=True)
    shown = remaining[:limit]

    for entity in shown:
        aka = ", ".join(
            a for a in entity.aliases if normalize_name(a) != normalize_name(entity.canonical_name)
        )
        line = f"{entity.entity_id}: {entity.canonical_name}"
        lines.append(f"{line} (aka {aka})" if aka else line)

    omitted = len(remaining) - len(shown)
    if omitted > 0:
        lines.append(f"...and {omitted} more known entities not shown here.")

    return "\n".join(lines)


# --- Idempotent per-document processing ----------------------------------------


def is_document_processed(registry: Registry, document_id: str, content_hash: str) -> bool:
    return any(
        doc.document_id == document_id and doc.content_hash == content_hash
        for doc in registry.processed_documents
    )

def _find_processed(registry: Registry, document_id: str) -> ProcessedDocument | None:
    for doc in registry.processed_documents:
        if doc.document_id == document_id:
            return doc
    return None

def remove_document_mentions(registry: Registry, document_id: str) -> None:
    kept_entities: list[RegistryEntity] = []
    for entity in registry.entities:
        entity.mentions = [m for m in entity.mentions if m.document_id != document_id]
        if entity.mentions or entity.source != "llm":
            kept_entities.append(entity)
    registry.entities = kept_entities
    registry.processed_documents = [
        doc for doc in registry.processed_documents if doc.document_id != document_id
    ]


def process_document(
    input_path: Path,
    corpus_root: Path,
    registry: Registry,
    lookup: list[LookupEntity],
    review_queue: list[dict],
    *,
    reprocess: bool = False,
) -> Path | None:
    document_text = input_path.read_text(encoding="utf-8")
    document_id = document_id_for(input_path, corpus_root)
    content_hash = content_hash_for(document_text)

    if is_document_processed(registry, document_id, content_hash):
        print(f"skip (unchanged): {document_id}")
        return None

    existing = _find_processed(registry, document_id)
    if existing is not None and existing.content_hash != content_hash:
        if not reprocess:
            print(f"skip (changed, use --reprocess to update): {document_id}", file=sys.stderr)
            return None
        remove_document_mentions(registry, document_id)

    known_context = build_known_entities_context(registry, lookup)
    entity_graph = graph.extract_entity_graph(
        document_text, document_id, known_entities_context=known_context
    )

    merge_document_graph(registry, entity_graph, document_id, lookup, review_queue)

    output_path = graph.default_output_path(input_path)
    output_path.write_text(
        json.dumps(entity_graph.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    registry.processed_documents.append(
        ProcessedDocument(
            document_id=document_id,
            content_hash=content_hash,
            processed_at=_now(),
            entity_graph_path=output_path.as_posix(),
            entity_count=len(entity_graph.entities),
            relationship_count=len(entity_graph.relationships),
        )
    )
    print(f"merged: {document_id} ({len(entity_graph.entities)} entities)")
    return output_path


def process_directory(
    root: Path,
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    lookup_path: Path = DEFAULT_LOOKUP_PATH,
    review_queue_path: Path = DEFAULT_REVIEW_QUEUE_PATH,
    corpus_root: Path | None = None,
    reprocess: bool = False,
) -> None:
    corpus_root = corpus_root or Path.cwd()
    registry = load_registry(registry_path)
    lookup = load_lookup(lookup_path)

    reconciled = reconcile_with_lookup(registry, lookup)
    if reconciled:
        print(f"reconciled {reconciled} entit{'y' if reconciled == 1 else 'ies'} via lookup.json")

    review_queue: list[dict] = []
    for path in sorted(root.glob(DOCUMENT_GLOB)):
        process_document(path, corpus_root, registry, lookup, review_queue, reprocess=reprocess)
        save_registry(registry, registry_path)

    if review_queue:
        existing_reviews: list[dict] = []
        if review_queue_path.exists():
            existing_reviews = json.loads(review_queue_path.read_text(encoding="utf-8"))
        review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        review_queue_path.write_text(
            json.dumps(existing_reviews + review_queue, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {len(review_queue)} ambiguous match(es) to {review_queue_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-document entity extractions into a corpus-wide registry."
    )
    parser.add_argument("directory", type=Path, help="Directory of *.JPG.txt files to process")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-merge documents whose .txt content changed since the last run",
    )
    args = parser.parse_args(argv)

    if not args.directory.exists():
        print(f"Directory not found: {args.directory}", file=sys.stderr)
        return 1

    process_directory(args.directory, reprocess=args.reprocess)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())