"""Prompt pool for a prose model's drafter (e.g. Hemmingway-1.5): English fiction, non-fiction and
rewriting, in the lengths and forms such a model is asked for. Only prompts come from datasets; the
target writes the answers later.

Categories: fiction, nonfiction, rewrite. Each record also carries `reasoning_effort`, rotated over
none/low/medium/xhigh, which generate.py passes through, so captures cover every thinking level.

Usage: build_prose_pool.py OUT.jsonl
"""
import itertools
import json
import random
import sys
from collections import Counter

from datasets import load_dataset

out = sys.argv[1]
rng = random.Random(41)
rows = []
EFFORTS = ["none", "low", "medium", "xhigh"]


def stream(path, split, **kw):
    try:
        return load_dataset(path, split=split, streaming=True, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e!r}"[:200], flush=True)
        return []


def take(src, cat, it, n, fn):
    c = 0
    try:
        for ex in it:
            p = fn(ex)
            if p:
                rows.append({"prompt": p, "source": src, "category": cat})
                c += 1
            if c >= n:
                break
    except Exception as e:  # noqa: BLE001
        print(f"{src}: stopped after {c}: {e!r}"[:200], flush=True)
    print(src, cat, c, flush=True)


def clean_wp(p):
    p = p.replace(" ,", ",").replace(" .", ".").replace(" '", "'").replace(" n't", "n't").replace("`` ", '"').replace(" ''", '"')
    for tag in ("[ WP ]", "[ EU ]", "[ CW ]", "[ TT ]", "[ PI ]", "[ RF ]", "[ MP ]", "[ IP ]", "[ OT ]"):
        p = p.replace(tag, "")
    return p.strip()


# ---------------------------------------------------------------- fiction
FIC_LEN = ["about 300 words", "400-700 words", "about 1,000 words", "1,500-2,000 words", "a complete short story"]
FIC_STYLE = ["", "", "Write it in the first person.", "Write it in the present tense.", "Keep the tone literary.",
             "Make it funny.", "Make it dark.", "Tell it mostly through dialogue.", "Write it for young readers."]


def wp_prompt(ex):
    p = clean_wp(ex.get("prompt") or "")
    if not 30 <= len(p) <= 600:
        return None
    style = rng.choice(FIC_STYLE)
    return f"{p}\n\nWrite a story ({rng.choice(FIC_LEN)}) based on this prompt.{(' ' + style) if style else ''}"


def wp_continue(ex):
    s = (ex.get("story") or "").replace("<newline>", "\n").strip()
    if len(s) < 2500:
        return None
    cut = s[:rng.randint(900, 1800)].rsplit(" ", 1)[0]
    return f"Continue this story from where it stops, in the same voice, for about {rng.choice(['500', '800', '1,200'])} words:\n\n{cut}"


wp = stream("euclaise/writingprompts", "train")
take("wp_prompt", "fiction", wp, 1300, wp_prompt)
take("wp_continue", "fiction", itertools.islice(stream("euclaise/writingprompts", "train"), 20000, None), 350, wp_continue)
take("dolly_creative", "fiction", stream("databricks/databricks-dolly-15k", "train"), 250,
     lambda ex: ex["instruction"] if ex["category"] == "creative_writing" and not ex["context"] and len(ex["instruction"]) >= 20 else None)

# ---------------------------------------------------------------- non-fiction
NF_FORM = [("an essay", "800-1,200 words"), ("a blog post", "600-900 words"), ("an opinion piece", "about 700 words"),
           ("a feature article", "1,000-1,500 words"), ("a short explainer for a general reader", "about 500 words"),
           ("a newsletter section", "about 400 words"), ("a speech", "about 600 words")]


def essay(ex):
    t = (ex.get("titles") or "").strip()
    if not 40 <= len(t) <= 500:
        return None
    t = t.removeprefix("The essay").strip()
    form, ln = rng.choice(NF_FORM)
    return f"Write {form} ({ln}) on this subject: the piece {t}"


take("essays", "nonfiction", stream("ChristophSchuhmann/essays-with-instructions", "train"), 700, essay)


def news(ex):
    h = (ex.get("highlights") or "").replace("\n", " ").strip()
    if not 80 <= len(h) <= 600:
        return None
    form, ln = rng.choice(NF_FORM[:5])
    return f"Write {form} ({ln}) about this story, for readers who have not heard of it:\n\n{h}"


take("news", "nonfiction", itertools.islice(stream("abisee/cnn_dailymail", "train", name="3.0.0"), 50000, None), 450, news)
ITEMS = ["a cast-iron skillet", "noise-cancelling headphones", "a hand-knitted wool scarf", "a standing desk", "a sourdough starter kit",
         "a used 2014 hatchback", "a weekend cabin rental", "a beginner's watercolour set", "a refurbished laptop", "a trail-running shoe",
         "a small-batch hot sauce", "a children's picture book", "a mechanical keyboard", "a vintage record player", "a yoga mat"]
EVERYDAY = ["Write a product description for {i}.", "Write an Amazon review of {i} that you have owned for a year.",
            "Write a short listing for selling {i} online.", "Write an email to a friend recommending {i}."]
take("everyday", "nonfiction", iter(range(10 ** 6)), 150,
     lambda _: rng.choice(EVERYDAY).format(i=rng.choice(ITEMS)))
take("no_robots_gen", "nonfiction", stream("HuggingFaceH4/no_robots", "train"), 450,
     lambda ex: ex["prompt"] if ex["category"] == "Generation" and 20 <= len(ex["prompt"]) <= 3000 else None)

# ---------------------------------------------------------------- rewriting
RW = ["Rewrite this so it reads as if a person wrote it, not an AI. Keep the meaning.",
      "Edit this for clarity and flow. Keep the author's points.",
      "Rewrite this as a casual blog post.", "Shorten this to half its length without losing the main points.",
      "Rewrite this for a general audience with no specialist background."]


def fineweb(ex):
    t = (ex.get("text") or "").strip()
    if len(t) < 1200:
        return None
    cut = t[:rng.randint(1200, 3500)].rsplit("\n", 1)[0] if "\n" in t[:3500] else t[:3000]
    return f"{rng.choice(RW)}\n\n{cut}"


take("fineweb_rewrite", "rewrite", stream("Lambent/creative-writing-2048-fineweb-edu-sample", "train"), 350, fineweb)
take("no_robots_rewrite", "rewrite", stream("HuggingFaceH4/no_robots", "train"), 250,
     lambda ex: ex["prompt"] if ex["category"] == "Rewrite" and 40 <= len(ex["prompt"]) <= 5000 else None)

seen, uniq = set(), []
for r in rows:
    if r["prompt"] not in seen:
        seen.add(r["prompt"]); uniq.append(r)
rng.shuffle(uniq)
for i, r in enumerate(uniq):
    r["id"] = 500000 + i
    r["reasoning_effort"] = EFFORTS[i % len(EFFORTS)]
with open(out, "w") as f:
    for r in uniq:
        f.write(json.dumps(r) + "\n")
print("total", len(uniq), dict(Counter(r["category"] for r in uniq)), dict(Counter(r["source"] for r in uniq)))
