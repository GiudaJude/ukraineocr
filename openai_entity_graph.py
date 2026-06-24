from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

OPENAI_MODEL_NER = os.getenv("OPENAI_MODEL_NER", "gpt-5.4-mini")
OPENAI_NER_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_NER_MAX_OUTPUT_TOKENS", "4000"))

SYSTEM_PROMPT = """ROLE: You are an expert paleographer specializing in 16th and 17th century
Lviv council records and Latin/Old Polish legal scripts. 

TASK: Your job is to extract canonical named entities and relationship triples from OCR text.
Return only schema-compliant JSON.
Do not include markdown, commentary, or explanatory prose.
"""

USER_PROMPT_TEMPLATE = """Extract canonical entities from the document below.

Entity types:
- person
- organization
- location
- date
- currency

Requirements:
- Canonicalize each entity to a stable, modernized label where possible.
- Keep mention_texts as they appear in the document.
- Use entity IDs like E1, E2, E3.
- Only include relationships whose head and tail both appear in the entities list.
- Relationships must use canonical entity IDs, not raw strings.
- If no entities or relationships are found, return empty arrays.

Document ID: {document_id}

Document text:
{document_text}
"""

ENTITY_GRAPH_SCHEMA = {
    "type": "json_schema",
    "name": "entity_graph_extraction",
    "description": "Canonical entity and relationship extraction from OCR text",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "document_id": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "entity_id": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": ["person", "organization", "location", "date", "currency"],
                        },
                        "canonical_name": {"type": "string"},
                        "mention_texts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "justification": {"type": "string"},
                    },
                    "required": [
                        "entity_id",
                        "entity_type",
                        "canonical_name",
                        "mention_texts",
                        "justification",
                    ],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "head_entity_id": {"type": "string"},
                        "relation": {"type": "string"},
                        "tail_entity_id": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": [
                        "head_entity_id",
                        "relation",
                        "tail_entity_id",
                        "evidence",
                    ],
                },
            },
        },
        "required": ["document_id", "entities", "relationships"],
    },
}


EntityType = Literal["person", "organization", "location", "date", "currency"]


class ExtractedEntity(BaseModel):
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    mention_texts: list[str] = Field(default_factory=list)
    justification: str


class EntityRelationship(BaseModel):
    head_entity_id: str
    relation: str
    tail_entity_id: str
    evidence: str


class EntityGraph(BaseModel):
    document_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your shell environment or .env file."
        )
    return OpenAI(api_key=api_key)


def build_prompt(document_text: str, document_id: str) -> str:
    return USER_PROMPT_TEMPLATE.format(document_id=document_id, document_text=document_text)


def default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(input_path.suffix + ".entity_graph.json")


def extract_entity_graph(document_text: str, document_id: str) -> EntityGraph:
    response = get_client().responses.create(
        model=OPENAI_MODEL_NER,
        instructions=SYSTEM_PROMPT,
        input=build_prompt(document_text, document_id),
        max_output_tokens=OPENAI_NER_MAX_OUTPUT_TOKENS,
        temperature=0.0,
        text={
            "format": ENTITY_GRAPH_SCHEMA,
            "verbosity": "low",
        },
    )

    return validate_entity_graph_json(response.output_text)


def validate_entity_graph_json(raw_json: str) -> EntityGraph:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc

    try:
        graph = EntityGraph.model_validate(parsed)
    except ValidationError as exc:
        raise RuntimeError(f"OpenAI returned schema-invalid JSON: {exc}") from exc

    entity_ids = {entity.entity_id for entity in graph.entities}
    for relationship in graph.relationships:
        if relationship.head_entity_id not in entity_ids:
            raise RuntimeError(
                f"Relationship head {relationship.head_entity_id!r} is missing from entities."
            )
        if relationship.tail_entity_id not in entity_ids:
            raise RuntimeError(
                f"Relationship tail {relationship.tail_entity_id!r} is missing from entities."
            )

    return graph


def run_file(input_path: Path, output_path: Path | None = None) -> Path:
    document_text = input_path.read_text(encoding="utf-8")
    document_id = input_path.name
    graph = extract_entity_graph(document_text, document_id)

    final_output_path = output_path or default_output_path(input_path)
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_output_path.write_text(
        json.dumps(graph.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return final_output_path


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or len(args) > 2:
        print("Usage: python openai_entity_graph.py <input.txt> [output.json]", file=sys.stderr)
        return 1

    input_path = Path(args[0])
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args[1]) if len(args) == 2 else None
    written_path = run_file(input_path, output_path)
    print(f"Wrote entity graph to: {written_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
