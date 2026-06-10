"""Champion-augmentation importer: load huikang's pre-built drill files as data.

/tmp/huikang_nemotron/augmentations/ holds 8463 drill .txt files written by the
champion's pipeline (matching 4515, splitting 1500, concatenation 1500,
spelling 648, lstrip 300). Each file is plain text in the exact format that
the champion's corpus.py consumes:

    [category]
    <category name>
    [prompt]
    <prompt text, may span many lines>
    [completion]
    <completion text, may span many lines>

This module reads those files purely AS DATA (no external code is executed)
and re-exposes them through the same generate() -> list[dict] interface as
the other augmenters in this package. Parsing mirrors the champion's own
corpus.py consumption byte-for-byte (first-occurrence splits, completion
rstrip of trailing newlines) so the imported prompt/completion text is
identical to what the champion trained on.

IDs are the file stems (8-hex, unique within the directory), so they are
stable across runs. Categories are the drill kinds verbatim.
"""

from __future__ import annotations

from pathlib import Path

AUGMENTATIONS_DIR = Path("/tmp/huikang_nemotron/augmentations")

# Ground truth measured over the directory (2026-06-10); generate() asserts these.
EXPECTED_COUNTS = {
    "matching": 4515,
    "splitting": 1500,
    "concatenation": 1500,
    "spelling": 648,
    "lstrip": 300,
}

_CATEGORY_MARK = "[category]\n"
_PROMPT_MARK = "\n[prompt]\n"
_COMPLETION_MARK = "\n[completion]\n"


def _parse_drill(text: str, name: str) -> tuple[str, str, str]:
    """Parse one drill file into (category, prompt, completion).

    Exactly replicates the champion corpus.py parsing:
        category   = text.split("[category]\\n", 1)[1].split("\\n[prompt]\\n", 1)[0]
        prompt     = text.split("[prompt]\\n", 1)[1].split("\\n[completion]\\n", 1)[0]
        completion = text.split("\\n[completion]\\n", 1)[1].rstrip("\\n")
    """
    if not text.startswith(_CATEGORY_MARK):
        raise ValueError(f"{name}: missing [category] header")
    if _PROMPT_MARK not in text or _COMPLETION_MARK not in text:
        raise ValueError(f"{name}: missing [prompt] or [completion] section")

    category = text.split(_CATEGORY_MARK, 1)[1].split(_PROMPT_MARK, 1)[0]
    prompt = text.split("[prompt]\n", 1)[1].split(_COMPLETION_MARK, 1)[0]
    completion = text.split(_COMPLETION_MARK, 1)[1].rstrip("\n")
    return category, prompt, completion


def generate(aug_dir: Path | str = AUGMENTATIONS_DIR) -> list[dict[str, str]]:
    """Import all champion drill files.

    Returns a list of dicts with keys id / category / prompt / completion,
    sorted by id (file stem). Raises if the directory is missing, any file is
    malformed/empty, ids collide, or per-category counts deviate from the
    known 4515/1500/1500/648/300 split.
    """
    aug_dir = Path(aug_dir)
    if not aug_dir.is_dir():
        raise FileNotFoundError(f"augmentations directory not found: {aug_dir}")

    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    counts: dict[str, int] = {}

    for path in sorted(aug_dir.glob("*.txt")):
        category, prompt, completion = _parse_drill(path.read_text(), path.name)

        if category not in EXPECTED_COUNTS:
            raise ValueError(f"{path.name}: unknown category {category!r}")
        if not prompt.strip():
            raise ValueError(f"{path.name}: empty prompt")
        if not completion.strip():
            raise ValueError(f"{path.name}: empty completion")

        drill_id = path.stem
        if drill_id in seen_ids:
            raise ValueError(f"duplicate drill id: {drill_id}")
        seen_ids.add(drill_id)
        counts[category] = counts.get(category, 0) + 1

        rows.append(
            {
                "id": drill_id,
                "category": category,
                "prompt": prompt,
                "completion": completion,
            }
        )

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"per-category counts {counts} != expected {EXPECTED_COUNTS}"
        )
    return rows
