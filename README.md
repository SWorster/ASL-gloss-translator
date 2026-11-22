# English to ASL gloss translation

`final.ipynb` translates English sentences into ASL gloss using the ASLG-PC12 corpus. It cleans and filters the corpus, then trains and compares several translators: two rule-based baselines, an LSTM, a Transformer and a fine-tuned T5. Each translator reads either the raw English or a spaCy dependency parse of it.

## Requirements

- macOS or Linux (on Windows, use WSL)
- Python 3.14
- `curl` and `tar`
- An internet connection for the first setup
- About 1.5 GB of disk space for setup: 1.2 GB for the Python packages, 300 MB for the T5 model, and 25 MB for the data, plus about 1.2 GB for the notebook's output once every model is trained

A GPU is optional. The notebook uses CUDA or an Apple GPU when one is available, but every model also trains on the CPU.

## Setup

Clone the repository and run the setup script from the project folder:

```bash
git clone https://github.com/SWorster/ASL-gloss-translator.git
cd ASL-gloss-translator
bash required.sh
```

`required.sh` sets up everything inside the project folder:

1. a virtual environment in `.venv`
2. the Python packages, at the versions the notebook was tested with (PyTorch, transformers, spaCy, pandas, sacreBLEU and matplotlib)
3. the spaCy English model (`en_core_web_sm`)
4. the ASLG-PC12 corpus, from [Hugging Face](https://huggingface.co/datasets/achrafothman/aslg_pc12), saved as `train.csv`
5. the [SCOWL](https://wordlist.aspell.net/scowl-readme/) English word lists, in `scowl-2020.12.07/`
6. the pretrained [flan-t5-small](https://huggingface.co/google/flan-t5-small) model, in the Hugging Face cache

The script skips any step that is already done, so it is safe to run again.

If your Python 3.14 isn't called `python3.14`, point the script to it:

```bash
PYTHON=/path/to/python3.14 bash required.sh
```

## Running the notebook

### VS Code

Open `final.ipynb`, choose **Select Kernel**, and pick the `.venv` environment in the project folder. Then choose **Run All**.

### Jupyter

The environment only includes the kernel, not Jupyter itself. Install Jupyter into it once, then start it:

```bash
.venv/bin/python -m pip install notebook
.venv/bin/jupyter notebook final.ipynb
```

## Before a full run

All settings are in the notebook's first cell. These are the ones you are most likely to change:

| Setting                                          | Default                  | What it does                                                                                       |
| ------------------------------------------------ | ------------------------ | -------------------------------------------------------------------------------------------------- |
| `FILTER_LIMIT`                                   | `None`                   | Use a random sample of this many pairs, for a quick test run. `None` uses the whole corpus.        |
| `EVAL_LIMIT`                                     | `None`                   | Score only a random sample of this many test pairs.                                                |
| `LSTM_DEVICE`, `TRANSFORMER_DEVICE`, `T5_DEVICE` | `"cpu"`, `"cpu"`, `None` | Where each model trains. `None` picks the fastest device available.                                |
| `REBUILD_CACHE`                                  | `False`                  | Filter the corpus again even if a cached result exists.                                            |
| `RETRAIN_MODELS`                                 | `False`                  | Train the neural models again even if saved models exist.                                          |
| `LSTM_EPOCHS`, `TRANSFORMER_EPOCHS`, `T5_EPOCHS` | `50`, `50`, `3`          | The most epochs each model trains for.                                                             |
| `EARLY_STOPPING_PATIENCE`                        | `3`                      | Stop after this many epochs without a new best validation loss, and keep the best epoch's weights. |

The neural models are slow on the full corpus. On an Apple M3 Max, training all nine took about 14½ hours: 22 to 83 minutes for each LSTM and 66 to 110 minutes for each Transformer on the CPU, and 30 minutes (raw English) to 4½ hours (flat parse) for each T5 on the Apple GPU. You can run the cells for one model at a time; restarting the kernel between models can make training noticeably faster. Each trained model is saved, so running its cell again loads it instead of training it, unless its settings or the data have changed. The comparison cells at the end only use the models that have been trained.

## Output

The notebook writes three folders:

- `cache/`: the filtered corpus and the trained models, so later runs skip filtering and training (about 1 GB once every model is trained)
- `data/interim/`: the kept, flagged and removed pairs as TSV files, with a summary
- `results/`: each translator's scores, predictions and (for the neural models) per-epoch losses and token accuracies, plus one combined `metrics.tsv` and `training_curves.png`, the LSTM and Transformer training curves

`examples.txt`, in the repository, holds every saved model's output on the notebook's example sentences and the training output of the final runs.

## Troubleshooting

- **`python3.14: command not found`**: install Python 3.14, or set `PYTHON` as shown under Setup.
- **The Apple GPU crashes during training**: `torch` is pinned to 2.9.1 because newer versions crash on the Apple GPU (MPS) under macOS 14. Don't upgrade it, or set the device to `"cpu"`.
- **spaCy can't find `en_core_web_sm`**: run `bash required.sh` again, then restart the notebook's kernel.
- **Results don't change after editing a filter rule**: increase `FILTER_RULES_VERSION` in the filtering cell, or set `REBUILD_CACHE = True`.
- **A model doesn't change after editing its code**: set `RETRAIN_MODELS = True`, or delete `cache/models/`. Saved models are only retrained automatically when their settings or the data change.
- **The training curves say "not recorded"**: the model was trained before the notebook recorded accuracy. Retrain it with `RETRAIN_MODELS = True`.
- **Training gets slower with each model**: restart the kernel between model cells. Saved models are reloaded instead of retrained, so no work is lost.

## Data sources

- **ASLG-PC12**: Othman, A. and Jemni, M. (2012). English-ASL Gloss Parallel Corpus 2012: ASLG-PC12. _5th Workshop on the Representation and Processing of Sign Languages, LREC 2012_, pp. 151–154.
- **SCOWL**: Spell Checker Oriented Word Lists, by Kevin Atkinson, version 2020.12.07.
- **flan-t5-small**: Chung et al. (2024). Scaling Instruction-Finetuned Language Models. _Journal of Machine Learning Research_ 25(70). Released by Google.
