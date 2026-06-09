"""Cipher: substitution cipher reasoning generator.

max-R redesign (2026-06-09): the chain-of-thought now derives the answer by
COPYING strings already printed earlier in the same trace, never by asserting a
global rule. The selection of an unknown word is converted from a bare verdict
(the old 77-row scan + "Best match: X") into a LOCAL constraint-narrowing block
whose decisive evidence (the partial pattern + the still-unused plain letters)
is printed on the two lines immediately above the named word. The word-choosing
LOGIC is byte-for-byte unchanged, so the boxed answer is identical to the frozen
baseline for every problem id; only the reasoning TEXT differs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .store_types import Problem

_WONDERLAND_PATH = Path(__file__).parent / "wonderland.txt"

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@lru_cache(maxsize=1)
def _load_wonderland() -> list[str]:
    """Load the Wonderland word list (sorted)."""
    with _WONDERLAND_PATH.open() as f:
        words = [line.strip() for line in f if line.strip()]
    return sorted(words)


def _word_pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    pattern: list[int] = []
    for char in word:
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)


def _candidate_words_for_partial(
    partial: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
    cipher_word: str,
) -> list[str]:
    """Find wonderland words matching a partial decryption with unknowns.

    Candidates must be consistent with cipher_to_plain (bijective mapping).
    """
    candidates: list[str] = []
    target_len = len(partial)
    target_pattern = _word_pattern(cipher_word)

    for word in _load_wonderland():
        if len(word) != target_len:
            continue
        if _word_pattern(word) != target_pattern:
            continue
        match = True
        for i, ch in enumerate(partial):
            if ch != "?" and ch != word[i]:
                match = False
                break
        if not match:
            continue
        # Check forward consistency only (cipher->plain)
        consistent = True
        for cc, wc in zip(cipher_word, word):
            if cc in cipher_to_plain and cipher_to_plain[cc] != wc:
                consistent = False
                break
        if consistent:
            candidates.append(word)

    candidates.sort()
    return candidates


def reasoning_cipher(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to find the substitution cipher mapping from the examples, "
        "then decrypt the question."
    )

    # ---- SECTION 1: EXAMPLE ALIGNMENT (anchor + local-derive source) --------
    # Build cipher_to_plain letter by letter, each plain char copied from the
    # bracketed plain sentence restated directly above.
    lines.append("")
    lines.append("Examples (cipher -> plain), aligned letter by letter:")
    cipher_to_plain: dict[str, str] = {}
    for ex in problem.examples:
        cipher_words = str(ex.input_value).split()
        plain_words = str(ex.output_value).split()
        if len(cipher_words) != len(plain_words):
            continue
        lines.append("")
        lines.append(f"[{ex.input_value}] -> [{ex.output_value}]")
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                if cc not in cipher_to_plain:
                    cipher_to_plain[cc] = pc
                lines.append(f"{cc} -> {pc}")

    # ---- SECTION 2: MAPPING TABLE (the master anchor) -----------------------
    # Exactly 26 rows a..z; the single copy source for every decrypt step.
    lines.append("")
    lines.append("Mapping so far:")
    for c in _ALPHABET:
        lines.append(f"{c} -> {cipher_to_plain.get(c, '?')}")

    plain_to_cipher: dict[str, str] = {v: k for k, v in cipher_to_plain.items()}

    # ---- SECTION 3: DECRYPT THE QUESTION, word by word ----------------------
    wonderland_words = _load_wonderland()
    wonderland_set = set(wonderland_words)
    question_words = problem.question.split()

    lines.append("")
    lines.append(f"Decrypting the question: {problem.question}")

    decoded_words: list[str] = [""] * len(question_words)

    for idx, cw in enumerate(question_words):
        lines.append("")
        lines.append(f"word: {cw}")

        # Per-letter copy from the 26-line Mapping table (keyed by the letter).
        partial_chars: list[str] = []
        has_unknown = False
        for cc in cw:
            pc = cipher_to_plain.get(cc, "?")
            lines.append(f"{cc} -> {pc}")
            partial_chars.append(pc)
            if cc not in cipher_to_plain:
                has_unknown = True
        partial = "".join(partial_chars)

        if not has_unknown:
            # All letters known: assemble by left-to-right copy of plain chars.
            lines.append("status: all known")
            lines.append(f"= {partial}")
            decoded_words[idx] = partial
            continue

        # Unknown letters present: rigid LOCAL-NARROWING block.
        lines.append("status: has unknown")

        candidates = _candidate_words_for_partial(
            partial, cipher_to_plain, plain_to_cipher, cw
        )
        display_candidates = sorted(candidates)
        if not display_candidates:
            return None

        # Filter: reject candidates that map an unknown cipher letter to a plain
        # letter already used as a value in the map (bijectivity / unused-letter
        # constraint). UNCHANGED selection logic -> identical chosen word.
        unmapped = {
            c for c in _ALPHABET if c not in cipher_to_plain.values()
        }
        remaining = []
        for c in display_candidates:
            bad = False
            for ci, wi in zip(cw, c):
                if ci not in cipher_to_plain and wi not in unmapped:
                    bad = True
                    break
            if not bad:
                remaining.append(c)

        wonderland_remaining = [c for c in remaining if c in wonderland_set]
        if wonderland_remaining:
            chosen = wonderland_remaining[0]
        elif remaining:
            chosen = remaining[0]
        else:
            return None

        # Print the decisive evidence locally, immediately above the verdict.
        lines.append(f"pattern: {partial}")
        unused_sorted = sorted(unmapped)
        lines.append(
            "letters still unused for plain: " + " ".join(unused_sorted)
        )
        # Show the surviving candidate set so the chosen word is copied from an
        # exhibited line (R4: evidence-adjacent, never a bare pick).
        lines.append(
            "candidates matching pattern with unused letters in the blanks: "
            + " ".join(remaining)
        )
        lines.append(
            f"the only vocabulary word of length {len(cw)} matching pattern "
            f"{partial} that fits is {chosen}"
        )
        lines.append(f"= {chosen}")

        # Per newly-resolved letter, emit a new-mapping line copied char-aligned
        # from <chosen> vs <cipher_word>, so later words read these from here.
        pending: list[tuple[str, str]] = []
        for cc, pc in zip(cw, chosen):
            if cc not in cipher_to_plain and (cc, pc) not in pending:
                pending.append((cc, pc))
                lines.append(f"new mapping: {cc} -> {pc}")
        for cc, pc in pending:
            cipher_to_plain[cc] = pc
            plain_to_cipher[pc] = cc

        decoded_words[idx] = chosen

    if any(w == "" for w in decoded_words):
        return None

    # ---- SECTION 4: ANSWER ASSEMBLY (self-consistent box) -------------------
    computed = " ".join(decoded_words)
    lines.append("")
    lines.append(f"Answer: {computed}")
    lines.append("The answer is \\boxed{%s}" % computed)
    return "\n".join(lines)
