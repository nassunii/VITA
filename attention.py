"""
Step 2. Build the `low` and `high` images.

Scores each word of the Introduction by the attention it receives in the last
layer of Llama-3-8B-Instruct, then colors the top 10% (`high`) or bottom 10%
(`low`) of words red. Layout is otherwise identical to the `black` image.

    texdataset/<category>/<id>_introduction_minimal_{low,high}.tex (+ .pdf)
    dataset/<category>/{low,high}/<id>_{low,high}.jpg
"""
import argparse
import glob
import os
import re

import torch
from transformers import AutoModel, AutoTokenizer

from arxiv2tex import compile_tex_to_pdf, find_poppler, pdf_to_jpg
from config import ATTENTION_MODEL, CATEGORIES, HIGHLIGHT_RATIO, TEX_DIR, image_path

MAX_TOKENS = 4096
LATEX_SPECIAL = set("\\{}$%&#^_~")
MATH_RE = re.compile(
    r"\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{(equation|align|gather|multline)\*?\}.*?\\end\{\1\*?\}",
    re.DOTALL,
)

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _tokenizer = AutoTokenizer.from_pretrained(ATTENTION_MODEL)
        _model = AutoModel.from_pretrained(
            ATTENTION_MODEL, dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
    return _tokenizer, _model


def split_minimal_tex(tex):
    """Split a minimal .tex (from arxiv2tex.py) into (preamble, intro body)."""
    preamble, rest = tex.split(r"\begin{document}", 1)
    body = rest.split(r"\section*{Introduction}", 1)[1].rsplit(r"\end{document}", 1)[0]
    return preamble, body.strip()


def candidate_words(body):
    """(start, end) spans of words that can be colored safely: plain text outside math, no LaTeX syntax."""
    math_spans = [m.span() for m in MATH_RE.finditer(body)]

    def in_math(s, e):
        return any(ms <= s and e <= me for ms, me in math_spans)

    spans = []
    for m in re.finditer(r"\S+", body):
        w = m.group()
        if any(c in LATEX_SPECIAL for c in w) or not re.search(r"[A-Za-z0-9]", w):
            continue
        if in_math(*m.span()):
            continue
        spans.append(m.span())
    return spans


@torch.no_grad()
def token_attention_scores(text):
    """
    Attention each token receives in the last layer, averaged over heads.
    With causal attention, token j can only be attended by queries i >= j, so we
    divide the column sum by (T - j); otherwise early tokens always win.
    Returns (scores[T], offsets[T]).
    """
    tokenizer, model = load_model()
    enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                    truncation=True, max_length=MAX_TOKENS)
    offsets = enc.pop("offset_mapping")[0].tolist()
    if enc["input_ids"].shape[1] >= MAX_TOKENS:
        print(f"[!] truncated to {MAX_TOKENS} tokens")

    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_attentions=True)
    att = out.attentions[-1][0].float().mean(dim=0)  # (T, T), averaged over heads
    del out

    T = att.shape[0]
    received = att.sum(dim=0) / torch.arange(T, 0, -1, device=att.device)
    return received.cpu(), offsets


def word_scores(body, spans):
    """Mean attention over the tokens that overlap each word span."""
    scores, offsets = token_attention_scores(body)
    result = []
    for s, e in spans:
        vals = [scores[i].item() for i, (ts, te) in enumerate(offsets)
                if te > ts and ts < e and te > s]  # skip special tokens (0, 0)
        result.append(sum(vals) / len(vals) if vals else None)
    return result


def color_words(body, spans, selected):
    out = body
    for idx in sorted(selected, key=lambda i: spans[i][0], reverse=True):
        s, e = spans[idx]
        out = out[:s] + r"{\color{red}" + out[s:e] + "}" + out[e:]
    return out


def make_highlighted_tex(tex_path):
    """Write the _low.tex and _high.tex versions of one minimal .tex. Returns {level: path}."""
    with open(tex_path, "r", encoding="utf-8") as f:
        preamble, body = split_minimal_tex(f.read())
    if r"\usepackage{xcolor}" not in preamble:
        preamble += "\\usepackage{xcolor}\n"

    spans = candidate_words(body)
    scores = word_scores(body, spans)
    scored = [i for i, s in enumerate(scores) if s is not None]  # words past the truncation point have no score
    if not scored:
        return {}
    ranked = sorted(scored, key=lambda i: scores[i])
    k = max(1, int(len(ranked) * HIGHLIGHT_RATIO))

    paths = {}
    for level, selected in [("low", ranked[:k]), ("high", ranked[-k:])]:
        tex = f"{preamble}\\begin{{document}}\n\\section*{{Introduction}}\n{color_words(body, spans, selected)}\n\\end{{document}}\n"
        path = tex_path[:-4] + f"_{level}.tex"
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        paths[level] = path
    return paths


def process_category(category, poppler_path, overwrite=False):
    tex_files = sorted(glob.glob(os.path.join(TEX_DIR, category, "*_introduction_minimal.tex")))
    print(f"[{category}] {len(tex_files)} tex files")
    for tex_file in tex_files:
        arxiv_id = os.path.basename(tex_file).split("_")[0]
        if not os.path.exists(image_path(category, "black", arxiv_id)):
            continue  # black image failed, keep the triplet consistent
        if not overwrite and all(os.path.exists(image_path(category, l, arxiv_id)) for l in ("low", "high")):
            continue
        try:
            for level, path in make_highlighted_tex(tex_file).items():
                pdf = compile_tex_to_pdf(path)
                if pdf and pdf_to_jpg(pdf, image_path(category, level, arxiv_id), poppler_path=poppler_path):
                    print(f"[{category}] {arxiv_id} {level}")
        except Exception as e:
            print(f"[ERROR] {tex_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    poppler = find_poppler()
    for cat in args.categories:
        process_category(cat, poppler, overwrite=args.overwrite)
