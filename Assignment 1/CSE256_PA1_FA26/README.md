# CSE 256 PA1: Deep Averaging Networks, Byte Pair Encoding, Skip-Gram

Nitya Dhagat (TSN 200483337)

## Requirements
Python 3.8+ with `torch`, `numpy`, `scikit-learn`, `matplotlib`.
CPU only is fine (no GPU needed).

Expected layout (run everything from the project root):

    data/train.txt
    data/dev.txt
    data/glove.6B.50d-relativized.txt
    data/glove.6B.300d-relativized.txt   # only needed for the --ablation 300d row

## Files
| File | Purpose |
|---|---|
| `main.py` | Entry point for all experiments |
| `DANmodels.py` | DAN model, word-level dataset, shared training loop (`train_model`), seeding |
| `BPEmodels.py` | BPE tokenizer (trained on `train.txt`), subword dataset, `SubwordDAN` |
| `BOWmodels.py`, `sentiment_data.py`, `utils.py` | Provided starter code (unchanged) |

## How to run (one command per part)

    python main.py --model BOW          # baseline Bag-of-Words (2- and 3-layer), 100 epochs
    python main.py --model DAN          # Part 1a/1b: DAN with GloVe vs random embeddings
    python main.py --model SUBWORDDAN   # Part 2: BPE DAN, vocab sizes 1k, 2k, 3k, 5k, 8k

Optional runs used in the report:

    python main.py --model DAN --ablation          # also runs the 1a hyperparameter grid (10 epochs/run)
    python main.py --model SUBWORDDAN --keep-eow   # keep the </w> marker in the token ids

Other options:

    --seeds N              seeds 0..N-1; results are mean +- std (default 3)
    --epochs N             epochs per run (default 20)
    --vocab-sizes 1000 3000 5000   BPE target vocabulary sizes
    --lr, --weight-decay, --batch-size, --eval-batch-size
    --embedding-dim, --hidden-size, --dropout     (SUBWORDDAN)
    --ablation-epochs N    epochs per ablation run (default 10)
    --train-file, --dev-file, --glove-50, --glove-300

## Default setup
- Loss: NLL on log-softmax outputs. Optimizer: Adam (lr 1e-3, weight decay 1e-5), batch size 64.
- DAN: mean of word embeddings (PAD masked out) -> Linear(50,100) -> ReLU -> Dropout(0.3) -> Linear(100,2).
- GloVe model: embeddings initialized from `glove.6B.50d-relativized.txt` and fine-tuned.
  Random models: embeddings ~ N(0, 0.1^2), trained from scratch.
- The epoch with the best dev accuracy is restored at the end of training
  (so "best dev accuracy" is selected on dev; the report also gives last-epoch accuracy).
- BPE: trained on `train.txt` only; `</w>` marks word ends and is stripped from token ids unless `--keep-eow`.
  Target vocabulary V = 115 base symbols + (V - 115) merges, plus PAD/UNK.
- All randomness is seeded (`set_seed`), so results are reproducible per seed.

## Outputs (written to the working directory)
- Plots: `train_accuracy.png`, `dev_accuracy.png` (BOW); `dan_dev_accuracy.png`, `dan_loss.png` (DAN);
  `bpe_<V>_dev_accuracy.png`, `bpe_<V>_train_accuracy.png`, `bpe_all_*_accuracy.png` (BPE)
- Results: `bow_results.json`, `dan_results.json`, `dan_ablation_results.json`,
  `subword_results.json`, `subword_results_keepeow.json`
- BPE diagnostics (merges, real word splits, vocabulary accounting): `bpe_diagnostics*.txt`

## Figures
Running the three commands writes these PNGs to the working directory. To keep them tidy, move them into `figures/` after a run:

    mkdir -p figures && mv *.png figures/

The report redraws the key curves (DAN comparison, BPE vocabulary sizes) directly from the `*_results.json` files, so the PNGs below are the raw outputs of the code.

### Baseline: Bag-of-Words (`--model BOW`)
| 2- vs 3-layer train accuracy | 2- vs 3-layer dev accuracy |
|---|---|
| ![BOW train accuracy](figures/train_accuracy.png) | ![BOW dev accuracy](figures/dev_accuracy.png) |

### Part 1: DAN, GloVe vs random embeddings (`--model DAN`)
Mean over seeds. `dan_dev_accuracy.png` shows GloVe, random (full vocab) and random (train vocab); `dan_loss.png` shows train and dev loss for GloVe and random (full vocab).

| Dev accuracy | Train/dev loss |
|---|---|
| ![DAN dev accuracy](figures/dan_dev_accuracy.png) | ![DAN loss](figures/dan_loss.png) |

### Part 2: SubwordDAN, BPE vocabulary sizes (`--model SUBWORDDAN`)
All vocabulary sizes on one plot (mean over seeds):

| Dev accuracy | Train accuracy |
|---|---|
| ![BPE dev accuracy, all sizes](figures/bpe_all_dev_accuracy.png) | ![BPE train accuracy, all sizes](figures/bpe_all_train_accuracy.png) |

One plot per vocabulary size:

| Size | Dev accuracy | Train accuracy |
|---|---|---|
| 1k | ![](figures/bpe_1000_dev_accuracy.png) | ![](figures/bpe_1000_train_accuracy.png) |
| 2k | ![](figures/bpe_2000_dev_accuracy.png) | ![](figures/bpe_2000_train_accuracy.png) |
| 3k | ![](figures/bpe_3000_dev_accuracy.png) | ![](figures/bpe_3000_train_accuracy.png) |
| 5k | ![](figures/bpe_5000_dev_accuracy.png) | ![](figures/bpe_5000_train_accuracy.png) |
| 8k | ![](figures/bpe_8000_dev_accuracy.png) | ![](figures/bpe_8000_train_accuracy.png) |

The `--keep-eow` run writes the same set with `_keepeow` in the name (for example `bpe_all_dev_accuracy_keepeow.png`, `bpe_5000_keepeow_dev_accuracy.png`).

## Results (dev set, mean over 3 seeds)
| Model | Best dev acc |
|---|---|
| BOW 2-layer / 3-layer (1 seed) | 0.737 / 0.742 |
| DAN, GloVe-50d fine-tuned | 0.801 |
| DAN, random init (full vocab / train vocab) | 0.791 / 0.796 |
| SubwordDAN BPE 1k / 2k / 3k / 5k / 8k | 0.751 / 0.762 / 0.781 / 0.795 / 0.799 |

## Approximate runtime (CPU)
BOW about 2 min; DAN main comparison about 1 min (about 5 more with `--ablation`); SUBWORDDAN about 3 min plus BPE training.

The report is `NityaDhagat_200483337_CSE256_PA1.pdf` (built from `report.tex`).