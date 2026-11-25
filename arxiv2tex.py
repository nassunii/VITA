"""
Step 1. Build the `black` images.

For each arXiv category: fetch recent paper IDs, download the LaTeX source,
extract the Introduction section, compile it into a one-column minimal PDF,
and save the first page as a JPEG.

    texdataset/<category>/<id>_introduction_minimal.tex (+ .pdf)
    dataset/<category>/black/<id>_intro.jpg
"""
import argparse
import gzip
import io
import os
import platform
import random
import re
import shutil
import subprocess
import tarfile
import time

import requests
from bs4 import BeautifulSoup
from pdf2image import convert_from_path

from config import CATEGORIES, DPI, TEX_DIR, image_path

HEADERS = {"User-Agent": "Mozilla/5.0 (VITA research crawler)"}
MIN_INTRO_CHARS = 50

PREAMBLE = r"""\documentclass{article}
\usepackage[a4paper, margin=0.7in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{xcolor}
\usepackage{url}
"""


# ---------------------------------------------------------------------------
# LaTeX cleaning
# ---------------------------------------------------------------------------
def remove_latex_comments(text):
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"(?<!\\)%.*", "", line)
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def remove_citations(text):
    """\\cite, \\citep, \\citet, \\citealp, ... -> (ref)"""
    return re.sub(r"\\cite[a-zA-Z]*\*?(\[[^\]]*\])*\{[^}]*\}", "(ref)", text)


def remove_floats(text):
    """Figures/tables reference files that are not in the minimal document."""
    for env in ["figure", "figure\\*", "table", "table\\*", "wrapfigure", "algorithm", "algorithm\\*"]:
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", text)
    return text


def remove_labels_and_refs(text):
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\(eq|auto|c|C)?ref\{[^}]*\}", "(ref)", text)
    return text


def clean_intro(text):
    text = remove_latex_comments(text)
    text = remove_floats(text)
    text = remove_citations(text)
    text = remove_labels_and_refs(text)
    return text.strip()


def extract_macros(tex):
    """Single-line \\newcommand / \\def definitions from the preamble, so custom macros still compile."""
    preamble = tex.split(r"\begin{document}")[0]
    macros = []
    for line in preamble.split("\n"):
        line = re.sub(r"(?<!\\)%.*", "", line).strip()
        if re.match(r"\\(newcommand|renewcommand|providecommand|def|DeclareMathOperator)\b", line):
            if line.count("{") == line.count("}") and "@" not in line:
                macros.append(line)
    return "\n".join(macros)


# ---------------------------------------------------------------------------
# Source download / introduction extraction
# ---------------------------------------------------------------------------
def flatten_tex(path, root, depth=0):
    """Inline \\input{...} and \\include{...} recursively."""
    if depth > 10 or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = remove_latex_comments(text)

    def repl(m):
        name = m.group(2).strip()
        candidates = [os.path.join(root, name), os.path.join(os.path.dirname(path), name)]
        for c in candidates:
            for p in (c, c + ".tex"):
                if os.path.isfile(p):
                    return flatten_tex(p, root, depth + 1)
        return ""

    return re.sub(r"\\(input|include)\{([^}]+)\}", repl, text)


INTRO_RE = re.compile(
    r"\\section\*?\s*\{\s*introduction\s*\}(.*?)(?=\\section\*?\s*\{|\\end\{document\}|\\appendix|\\bibliography\{|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def extract_introduction_from_tex(tex_content):
    """Return the cleaned Introduction body, or None."""
    m = INTRO_RE.search(tex_content)
    if not m:
        return None
    intro = clean_intro(m.group(1))
    return intro if len(intro) >= MIN_INTRO_CHARS else None


def get_recent_arxiv_ids(category="cs"):
    url = f"https://arxiv.org/list/{category}/recent?skip=0&show=500"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] {category}: {e}")
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    tags = soup.find_all("a", href=re.compile(r"^/abs/\d{4}\.\d{4,5}(v\d+)?$"))
    return sorted({t["href"].split("/")[-1] for t in tags})


def download_source(arxiv_id, work_dir):
    """Download and unpack the e-print. Returns the directory with .tex files, or None."""
    r = requests.get(f"https://arxiv.org/e-print/{arxiv_id}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.content
    os.makedirs(work_dir, exist_ok=True)

    if data[:4] == b"%PDF":
        return None  # no LaTeX source available

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            members = [m for m in tar.getmembers()
                       if m.isfile() and not m.name.startswith("/") and ".." not in m.name]
            tar.extractall(work_dir, members=members)
        return work_dir
    except tarfile.TarError:
        pass

    # Single gzipped .tex file
    try:
        text = gzip.decompress(data)
    except OSError:
        text = data
    with open(os.path.join(work_dir, "main.tex"), "wb") as f:
        f.write(text)
    return work_dir


def find_introduction(src_dir):
    """Look through every .tex file (main files first) for an Introduction section."""
    tex_files = []
    for root, _, files in os.walk(src_dir):
        tex_files += [os.path.join(root, f) for f in files if f.endswith(".tex")]

    def is_main(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return r"\documentclass" in f.read()

    mains = [p for p in tex_files if is_main(p)]
    for path in mains + [p for p in tex_files if p not in mains]:
        flat = flatten_tex(path, src_dir)
        intro = extract_introduction_from_tex(flat)
        if intro:
            macros = extract_macros(flat) if path in mains else ""
            return intro, macros
    return None, None


# ---------------------------------------------------------------------------
# TeX -> PDF -> JPG (shared with attention.py)
# ---------------------------------------------------------------------------
def build_minimal_tex(body, macros=""):
    return f"""{PREAMBLE}{macros}

\\begin{{document}}
\\section*{{Introduction}}
{body}
\\end{{document}}
"""


def compile_tex_to_pdf(tex_path, pdflatex_path=None):
    """Compile in the .tex file's directory. Returns the PDF path, or None on failure."""
    pdflatex_path = pdflatex_path or shutil.which("pdflatex") or "pdflatex"
    tex_dir = os.path.dirname(os.path.abspath(tex_path))
    tex_name = os.path.basename(tex_path)
    pdf_path = os.path.join(tex_dir, tex_name[:-4] + ".pdf")
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    # nonstopmode: minor errors (e.g. an undefined custom macro) still produce a PDF
    subprocess.run(
        [pdflatex_path, "-interaction=nonstopmode", tex_name],
        cwd=tex_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for ext in (".aux", ".out"):
        p = pdf_path[:-4] + ext
        if os.path.exists(p):
            os.remove(p)

    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        return pdf_path
    print(f"[!] pdflatex failed: {tex_path} (see {tex_name[:-4]}.log)")
    return None


def find_poppler():
    """Return the Poppler bin directory on Windows; None elsewhere (pdf2image then uses PATH)."""
    if platform.system() != "Windows":
        return None
    for path in [os.environ.get("POPPLER_PATH", "")] + os.environ.get("PATH", "").split(os.pathsep):
        if path and os.path.exists(os.path.join(path, "pdftoppm.exe")):
            return path
    raise FileNotFoundError("Poppler not found. Install it or add its bin/ directory to PATH.")


def pdf_to_jpg(pdf_path, output_path, dpi=DPI, poppler_path=None):
    """Save the first page of the PDF as a JPEG."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path,
                               first_page=1, last_page=1)
    if not images:
        return False
    images[0].convert("RGB").save(output_path, "JPEG")
    return True


# ---------------------------------------------------------------------------
def collect_category(category, num_papers, poppler_path, temp_dir="temp_arxiv", seed=0):
    ids = get_recent_arxiv_ids(category)
    print(f"[{category}] {len(ids)} recent papers")
    random.Random(seed).shuffle(ids)

    tex_dir = os.path.join(TEX_DIR, category)
    os.makedirs(tex_dir, exist_ok=True)

    done = sum(os.path.exists(image_path(category, "black", i)) for i in ids)
    for arxiv_id in ids:
        if done >= num_papers:
            break
        if os.path.exists(image_path(category, "black", arxiv_id)):
            continue

        work_dir = os.path.join(temp_dir, arxiv_id)
        try:
            src = download_source(arxiv_id, work_dir)
            if not src:
                continue
            intro, macros = find_introduction(src)
            if not intro:
                continue

            tex_path = os.path.join(tex_dir, f"{arxiv_id}_introduction_minimal.tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(build_minimal_tex(intro, macros))

            pdf_path = compile_tex_to_pdf(tex_path)
            if pdf_path and pdf_to_jpg(pdf_path, image_path(category, "black", arxiv_id), poppler_path=poppler_path):
                done += 1
                print(f"[{category}] {done}/{num_papers} {arxiv_id}")
        except Exception as e:
            print(f"[ERROR] {arxiv_id}: {e}")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            time.sleep(3)  # arXiv asks for at most one request every 3 seconds

    print(f"[{category}] collected {done} papers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--num_papers", type=int, default=5, help="papers per category")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    poppler = find_poppler()
    for cat in args.categories:
        collect_category(cat, args.num_papers, poppler, seed=args.seed)
