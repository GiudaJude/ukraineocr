from __future__ import annotations

import csv
import os
import re
import sys

from dotenv import load_dotenv
from collections import defaultdict
from pathlib import Path

load_dotenv()

WORD_PATTERN = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+")
MIN_WORD_LEN = 4
POLISH_LETTERS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

RARE_MAX = int(os.getenv("CONSISTENCY_RARE_MAX", "2"))
COMMON_MIN = int(os.getenv("CONSISTENCY_COMMON_MIN", "3"))
MAX_DISTANCE_RATIO = float(os.getenv("CONSISTENCY_MAX_DISTANCE_RATIO", "0.3"))

def levenshtein(word1: str, word2: str) -> int:
    if word1 == word2:
        return 0
    if not word1:
        return len(word2)
    if not word2:
        return len(word1)
    prev = list(range(len(word2) + 1))
    for i, word1_letter in enumerate(word1, start = 1):
        current = [i] + [0] * len(word2)
        for j, word2_letter in enumerate(word2, start=1):
            cost = 0 if word1_letter == word2_letter else 1
            current[j] = min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + cost)
        prev = current
    return prev[-1]

def iter_tokens(path: Path) -> list[tuple[str, int]]:
    tokens = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in WORD_PATTERN.finditer(line):
            token = match.group(0)
            if len(token) >= MIN_WORD_LEN:
                tokens.append((token, line_number))
    return tokens

def collect_corpus(target: Path, pattern: str = "*.parsed.txt") -> dict[str, list[tuple[str, str, int]]]:
    files = [target] if target.is_file() else sorted(target.glob(pattern))
    occurences: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for file_path in files:
        for token, line_number in iter_tokens(file_path):
            occurences[token.lower()].append((token, file_path.name, line_number))
    return occurences

def looks_polish(token:str) -> bool:
    lower = token.lower()
    return any(character in POLISH_LETTERS for character in token) or any(
        d in lower for d in ("sz", "cz", "rz", "dz")
    )

def find_outliers( occurences: dict[str, list[tuple[str, str, int]]]) -> tuple[list[dict], list[dict]]:
    counts = {key: len(v) for key, v in occurences.items()}
    rare_keys = [k for k, c in counts.items() if c <= RARE_MAX]
    common_keys = [k for k, c in counts.items() if c >= COMMON_MIN]

    common_by_length: dict[int, list[str]] = defaultdict(list)
    for key in common_keys:
        common_by_length[len(key)].append(key)

    matched, unmatched = [], []
    for rare in rare_keys:
        threshold = max(2, int(len(rare) * MAX_DISTANCE_RATIO))
        best_match, best_distance = None, None
        for length in range(len(rare) - threshold, len(rare) + threshold + 1):
            for common in common_by_length.get(length, []):
                distance = levenshtein(rare, common)
                if distance <= threshold and (best_distance is None or distance < best_distance):
                    best_match, best_distance = common, distance

        exact_spelling = occurences[rare][0][0]
        locations = "; ".join(f"{f}:{ln}" for _, f, ln in occurences[rare])
        row = {
            "spelling": exact_spelling,
            "count": counts[rare],
            "looks_polish": looks_polish(exact_spelling),
            "locations": locations,
        }
        if best_match:
            row.update(
                {
                    "suggested_correction": occurences[best_match][0][0],
                    "correction_count": counts[best_match],
                    "edit_distance": best_distance,
                }
            )
            matched.append(row)
        else:
            unmatched.append(row)
    return matched, unmatched

def write_csv(rows: list[dict], fieldnames: list[str], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("Usage: pythong post_tokenization.py <file-or-directory>", file=sys.stderr)
        return 1

    target = Path(args[0])
    occurances = collect_corpus(target)
    matched, unmatched = find_outliers(occurances)

    out_dir = target.parent if target.is_file() else target
    matched_path = out_dir / "likely_misreads.csv"
    unmatched_path = out_dir / "unmatched_rarities.csv"

    write_csv(
        sorted(matched, key=lambda r: -r["correction_count"]),
        ["spelling", "count", "suggested_correction", "correction_count",
         "edit_distance", "looks_polish", "locations"],
        matched_path
    )
    write_csv(
        sorted(unmatched, key=lambda r: r["spelling"]),
        ["spelling", "count", "looks_polish", "locations"],
        unmatched_path,
    )

    print(f"{len(matched)} likely misreads -> {matched_path}")
    print(f"{len(unmatched)} unmatched rarities (review manually) -> {unmatched_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
