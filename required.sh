#!/usr/bin/env bash
# Sets up everything final.ipynb needs, from scratch, in this folder:
# a virtual environment with the Python packages, the spaCy English model,
# the ASLG-PC12 data (train.csv), the SCOWL word lists, and the pretrained
# T5 checkpoint.
# Steps whose result already exists are skipped, so it is safe to re-run.
#
# Usage, from anywhere: bash required.sh
# Tested with Python 3.14 on macOS 14.
# Set PYTHON to use another interpreter, e.g. PYTHON=python3.13 bash required.sh
set -euo pipefail

# Work in the folder this script is in, wherever it is run from.
cd "$(dirname "$0")"

# 1. A virtual environment in .venv; every later step uses its Python.
PYTHON="${PYTHON:-python3.14}"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
VENV_PYTHON=".venv/bin/python"

# 2. The Python packages, at the versions the notebook was tested with.
# torch is pinned to 2.9.1: newer versions crash on the Apple GPU (MPS) under
# macOS 14.
# pyarrow reads the ASLG-PC12 parquet file in step 4.
# matplotlib draws the training curves.
# ipykernel lets Jupyter or VS Code run the notebook with this environment.
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install \
    "torch==2.9.1" \
    "transformers==5.22.1" \
    "sentencepiece==0.2.2" \
    "pandas==3.0.6" \
    "pyarrow==26.0.1" \
    "spacy==3.8.18" \
    "sacrebleu==2.6.0" \
    "matplotlib==3.11.2" \
    "ipykernel"

# 3. The spaCy English model, unless it is already installed.
# The check exits with 0 when the model is installed, so the download only
# runs when it is missing.
"$VENV_PYTHON" -c \
    "import spacy, sys; sys.exit(not spacy.util.is_package('en_core_web_sm'))" \
    || "$VENV_PYTHON" -m spacy download en_core_web_sm

# 4. ASLG-PC12 (87,710 English-gloss pairs) from Hugging Face, saved as
# train.csv with a "gloss,text" header, the format the notebook reads.
if [ ! -f train.csv ]; then
    curl -fL -o aslg_pc12.parquet \
        "https://huggingface.co/datasets/achrafothman/aslg_pc12/resolve/main/data/train-00000-of-00001.parquet"
    "$VENV_PYTHON" -c \
        "import pandas as pd; pd.read_parquet('aslg_pc12.parquet').to_csv('train.csv', index=False)"
    rm aslg_pc12.parquet
fi

# 5. The SCOWL word lists (version 2020.12.07) for the non-English filter,
# unpacked into scowl-2020.12.07/.
if [ ! -d scowl-2020.12.07 ]; then
    curl -fL -o scowl.tar.gz \
        "https://downloads.sourceforge.net/project/wordlist/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz"
    tar -xzf scowl.tar.gz
    rm scowl.tar.gz
fi

# 6. The pretrained T5 checkpoint (about 300 MB), into the Hugging Face cache,
# so the T5 cells don't have to download it.
"$VENV_PYTHON" -c "
from transformers import AutoTokenizer, T5ForConditionalGeneration
AutoTokenizer.from_pretrained('google/flan-t5-small')
T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')
"

echo "Done. Open final.ipynb with the .venv kernel and run all cells."
