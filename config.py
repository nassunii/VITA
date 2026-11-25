import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEX_DIR = os.path.join(BASE_DIR, "texdataset")      # texdataset/<category>/<id>_introduction_minimal.tex
DATASET_DIR = os.path.join(BASE_DIR, "dataset")     # dataset/<category>/{black,low,high}/<id>_*.jpg

CATEGORIES = ["cs", "math", "stat", "econ", "eess", "physics", "q-bio", "q-fin"]

ATTENTION_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HIGHLIGHT_RATIO = 0.1
DPI = 300


def image_path(category, level, arxiv_id):
    suffix = "intro" if level == "black" else level
    return os.path.join(DATASET_DIR, category, level, f"{arxiv_id}_{suffix}.jpg")
