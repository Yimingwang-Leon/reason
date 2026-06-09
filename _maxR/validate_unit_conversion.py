"""Validation for the unit_conversion max-R redesign."""
import json
import re

from tokenizers import Tokenizer

from src.problems import load_problems
from src.reasoning import extract_answer
from src.reasoners.unit_conversion import reasoning_unit_conversion

GEN_LIMIT = 7680

baseline = json.load(open("_maxR/baseline_unit_conversion.json"))
probs = {p.id: p for p in load_problems() if p.category == "unit_conversion"}
tok = Tokenizer.from_file("src/tokenizer.json")


def box_of(p):
    cot = reasoning_unit_conversion(p)
    if cot is None:
        return None, None
    return cot, extract_answer(cot)


# ---- (1) BYTE-IDENTICAL BOX GATE ----
n_checked = 0
n_mismatch = 0
mismatch_ids = []
for pid, base in baseline.items():
    n_checked += 1
    p = probs.get(pid)
    if p is None:
        n_mismatch += 1
        if len(mismatch_ids) < 5:
            mismatch_ids.append(pid + " (missing problem)")
        continue
    cot, box = box_of(p)
    # reasoner returns None -> box null; baseline null matches
    got = None if (cot is None or box == "") else box
    exp = base  # may be None (null)
    if got != exp:
        n_mismatch += 1
        if len(mismatch_ids) < 5:
            mismatch_ids.append(f"{pid}: got={got!r} exp={exp!r}")

# also verify no ids in corpus that are absent from baseline
extra = [pid for pid in probs if pid not in baseline]

# ---- (2) TOKEN CAP ----
max_tokens = 0
max_tok_id = None
for pid, p in probs.items():
    cot, box = box_of(p)
    if cot is None:
        box = baseline.get(pid)
        if box is None:
            continue
    completion = f"{cot}\n</think>\n\\boxed{{{box}}}<|im_end|>"
    n = len(tok.encode(completion).ids)
    if n > max_tokens:
        max_tokens = n
        max_tok_id = pid
under_cap = max_tokens <= GEN_LIMIT

# ---- (3) DETERMINISM ----
import random
sample_ids = sorted(probs.keys())
random.Random(0).shuffle(sample_ids)
sample_ids = sample_ids[:3]
deterministic = True
for pid in sample_ids:
    a = reasoning_unit_conversion(probs[pid])
    b = reasoning_unit_conversion(probs[pid])
    if a != b:
        deterministic = False

# ---- (4) GREP_CLEAN: no global-rule-assertion phrasing ----
GLOBAL_PATTERNS = [
    r"the rule is",
    r"the pattern is",
    r"the operation is",
    r"\bsecret\b",
    r"the shift",
    r"the formula",
]
grep_hits = []
for pid, p in probs.items():
    cot, _ = box_of(p)
    if cot is None:
        continue
    low = cot.lower()
    for pat in GLOBAL_PATTERNS:
        if re.search(pat, low):
            grep_hits.append((pid, pat))
            break
    # bare "Best:" without printed comparisons
    if re.search(r"^\s*best\s*:", low, re.MULTILINE):
        grep_hits.append((pid, "bare Best:"))
grep_clean = len(grep_hits) == 0

# ---- (5) BOX-COPY self-consistency: boxed appears on an earlier '= {box}' line
box_copy_fail = 0
for pid, p in probs.items():
    cot, box = box_of(p)
    if cot is None or box == "":
        continue
    if f"= {box}\n" not in cot + "\n":
        box_copy_fail += 1

# ---- (6) BRACE in answer ----
brace_fail = 0
for pid, p in probs.items():
    _, box = box_of(p)
    if box and ("{" in box or "}" in box):
        brace_fail += 1

print("=" * 50)
print(f"n_checked={n_checked}  n_mismatch={n_mismatch}")
print(f"sample_mismatch_ids={mismatch_ids}")
print(f"extra_ids_not_in_baseline={extra[:5]} (total {len(extra)})")
print(f"max_completion_tokens={max_tokens} (id={max_tok_id})  under_cap={under_cap}")
print(f"deterministic={deterministic} (sampled {sample_ids})")
print(f"grep_clean={grep_clean}  grep_hits={grep_hits[:5]}")
print(f"box_copy_fail={box_copy_fail}")
print(f"brace_fail={brace_fail}")
print("=" * 50)

import sys
ok = (n_mismatch == 0 and len(extra) == 0 and under_cap and deterministic
      and grep_clean and box_copy_fail == 0 and brace_fail == 0)
print("ALL_PASS=" + str(ok))
sys.exit(0 if ok else 1)
