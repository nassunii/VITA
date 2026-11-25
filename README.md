# VITA: Visual In-image Text Analysis on Vision-Language Models

We render the Introduction of recent arXiv papers as images in three conditions:

| Condition | Description |
|---|---|
| `black` | Original text |
| `high` | Top 10% words by attention score colored red |
| `low` | Bottom 10% words by attention score colored red |

Word-level attention scores are computed with Llama-3-8B-Instruct (last layer, averaged over heads).

## Installation

```bash
git clone https://github.com/nassunii/VITA.git
cd VITA
pip install -r requirements.txt
```

External dependencies:
- `pdflatex` (TeX Live or MiKTeX)
- Poppler (`brew install poppler`, `apt install poppler-utils`; on Windows, add its `bin/` to `PATH` or set `POPPLER_PATH`)

Access to [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) is required for attention scoring.

## Usage

```bash
# 1. Collect papers and render black images
python arxiv2tex.py --num_papers 5

# 2. Render high / low images
python attention.py
```

Both scripts accept `--categories`. Existing outputs are skipped. Paths and settings are in `config.py`.

ArXiv categories: `cs`, `math`, `stat`, `econ`, `eess`, `physics`, `q-bio`, `q-fin`.

## Pipeline

**`arxiv2tex.py`** samples papers from `arxiv.org/list/<category>/recent`, downloads the LaTeX source, and extracts the Introduction section. Citations and references are replaced with `(ref)`, and figures and tables are removed. The text is compiled into a single-column document and the first page is saved at 300 DPI.

**`attention.py`** computes the attention each token receives in the last layer, normalized by the number of positions that can attend to it under causal masking. Token scores are averaged per word. Only plain-text words outside math mode are eligible for coloring. The colored text is compiled with the same layout as the `black` image.

## Outputs

```
texdataset/<category>/<id>_introduction_minimal{,_low,_high}.tex
dataset/<category>/black/<id>_intro.jpg
dataset/<category>/low/<id>_low.jpg
dataset/<category>/high/<id>_high.jpg
```

## Repository Structure

```
├── config.py        # paths, categories, attention settings
├── arxiv2tex.py     # paper collection, black images
└── attention.py     # attention scoring, high/low images
```
