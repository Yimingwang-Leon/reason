"""Categorize train.csv into the 9 canonical sub-tasks and dump structural stats."""
import pandas as pd
import re
from collections import Counter

pd.set_option("display.max_colwidth", 500)
pd.set_option("display.width", 240)


def classify(p: str) -> str:
    pl = p.lower()
    if "secret bit manipulation rule transforms 8-bit binary" in pl:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in pl:
        return "cipher"
    if "secret unit conversion is applied to measurements" in pl:
        return "unit_conversion"
    if "gravitational constant has been secretly changed" in pl:
        return "gravity"
    if "numbers are secretly converted into a different numeral system" in pl:
        return "numeral"
    if "secret set of transformation rules is applied to equations" in pl:
        body = p.split("Below are a few examples:")[-1].split("Now,")[0]
        is_symbolic = not bool(re.search(r"\d", body))
        now = p.split("Now,")[-1].lower()
        is_guess = any(t in now for t in (
            "what input", "find the input", "what value", "find the value",
            "guess", "what would produce", "what equation", "expression that"))
        if is_symbolic:
            return "cryptarithm_guess" if is_guess else "cryptarithm_deduce"
        return "equation_numeric_guess" if is_guess else "equation_numeric_deduce"
    return "other"


df = pd.read_csv("data/train.csv")
df["answer"] = df["answer"].astype(str)
df["cat"] = df["prompt"].apply(classify)
df["plen"] = df["prompt"].str.len()
df["alen"] = df["answer"].str.len()

print("=" * 78)
print("CATEGORY DISTRIBUTION")
print("=" * 78)
print(df["cat"].value_counts().sort_index())
print(f"Total: {len(df)}   (target = 9500)")

print("\n" + "=" * 78)
print("LENGTH & DIVERSITY PER CATEGORY")
print("=" * 78)
agg = df.groupby("cat").agg(
    n=("id", "count"),
    plen_mean=("plen", "mean"),
    plen_max=("plen", "max"),
    alen_mean=("alen", "mean"),
    uniq_ans=("answer", "nunique"),
)
print(agg.round(1))

print("\n" + "=" * 78)
print("SAMPLE PROMPT + ANSWER PER CATEGORY")
print("=" * 78)
for cat in sorted(df["cat"].unique()):
    s = df[df["cat"] == cat].iloc[0]
    print(f"\n--- {cat} ({len(df[df['cat']==cat])} rows, id={s['id']}, answer={s['answer']!r}) ---")
    print(s["prompt"])

print("\n" + "=" * 78)
print("ANSWER CHARACTER SETS")
print("=" * 78)
for cat in sorted(df["cat"].unique()):
    chars = Counter()
    for a in df[df["cat"] == cat]["answer"]:
        chars.update(a)
    print(f"  {cat:25s} unique={len(chars):3d}  top: "
          f"{', '.join(f'{c!r}:{n}' for c, n in chars.most_common(8))}")

print("\n" + "=" * 78)
print("STRUCTURAL DIVERSITY (hidden constants / mappings)")
print("=" * 78)
# gravity: secret g = 2*d / t^2 from first example
grav = df[df["cat"] == "gravity"].copy()
grav["g"] = grav["prompt"].apply(
    lambda p: (lambda m: 2 * float(m[0][1]) / float(m[0][0]) ** 2 if m else None)
    (re.findall(r"For t = ([\d.]+)s, distance = ([\d.]+)", p))
)
print(f"  gravity g           : "
      f"{grav['g'].min():.2f} – {grav['g'].max():.2f}  "
      f"(distinct≈{grav['g'].round(0).nunique()})")
# unit_conv factor
uc = df[df["cat"] == "unit_conversion"].copy()
uc["factor"] = uc["prompt"].apply(
    lambda p: (lambda m: float(m[0][1]) / float(m[0][0]) if m else None)
    (re.findall(r"([\d.]+) m becomes ([\d.]+)", p))
)
print(f"  unit_conv factor    : "
      f"{uc['factor'].min():.3f} – {uc['factor'].max():.3f}  "
      f"(distinct≈{uc['factor'].round(2).nunique()})")
# numeral: target system char set
num = df[df["cat"] == "numeral"].copy()
num["sys"] = num["answer"].apply(lambda a: "".join(sorted(set(a))))
print(f"  numeral systems     : {num['sys'].nunique()} distinct char-sets")
print(f"    top 5: {num['sys'].value_counts().head(5).to_dict()}")

# bit_manipulation: how often the answer is 0/255/just rotation of input
bm = df[df["cat"] == "bit_manipulation"]
print(f"  bit_manip answer=0x00 / 0xFF : "
      f"{(bm['answer']=='00000000').sum()} / {(bm['answer']=='11111111').sum()}")

# Save categorized
df[["id", "cat", "prompt", "answer"]].to_csv("data/train_cat.csv", index=False)
print(f"\n[saved] data/train_cat.csv ({len(df)} rows)")
