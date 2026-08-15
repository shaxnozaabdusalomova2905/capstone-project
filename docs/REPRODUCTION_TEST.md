# Reproduction Test

How to check that this project works on your own machine, and what you should see.

---

## The quickest check — about one minute

Open `notebooks/05_inference_demo.ipynb` in Google Colab and run the first cell.

In a completely fresh session with nothing prepared, it clones this repository, loads the trained model, and is ready to classify images. You should see:

```
Cloning https://github.com/shaxnozaabdusalomova2905/capstone-project.git ...
Repository : /content/capstone-project
Weights    : resnet18_fold2.pt (44.8 MB)
Device     : cuda
Input size : 224x224 RGB
Classes    : 9 - Broken stitch, Needle mark, Pinched fabric, Vertical, defect free, hole, horizontal, lines, stain
```

If that appears, everything needed is present and correct. No dataset download, no training, no files to transfer.

---

## Even quicker — no installation at all

Open the live app: **https://capstone-project-fnrnvvgboe6b2i8gczg9ju.streamlit.app**

Pick a sample image or upload your own. It runs the same model from the same weights in this repository.

---

## Running it on your own computer

Tested on macOS with Python 3.9. Roughly ten minutes, most of it downloading PyTorch.

```bash
git clone https://github.com/shaxnozaabdusalomova2905/capstone-project.git
cd capstone-project
```

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows the activate line is `.venv\Scripts\activate`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This downloads about 500 MB. `requirements.txt` points at the CPU build of PyTorch, so it will not try to fetch a 2 GB GPU version.

**Then either start the web app:**

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`.

**Or classify from the command line:**

```bash
python src/predict.py assets/demo/stain.jpg
```

Expected shape of the output:

```
model: resnet18_fold2.pt on cpu

stain.jpg
  stain  (99.2%)   [DEFECT]
     also considered: hole (0.4%)
```

---

## What you should get

| Check | Expected |
|---|---|
| Model loads | `resnet18_fold2.pt`, 44.8 MB |
| Classes found | 9 |
| A clear defect image | correctly classified, high confidence |
| A blank or black image | refused, not classified |
| A text file renamed `.jpg` | refused with a clear message |

The refusals are as important as the predictions. A model will answer confidently on almost any input; refusing what it cannot sensibly handle is deliberate.

---

## Reproducing the training results

Only needed if you want to verify the numbers rather than use the model. Requires a GPU — expect about 90 minutes in total.

| Notebook | What it does | Time | Key result |
|---|---|---|---|
| `01_dataset_exploration.ipynb` | Explores the raw data | ~5 min | 3,067 files, 330 of them duplicates |
| `02_data_preprocessing.ipynb` | Cleans and splits | ~10 min | 2,737 images, 5 folds |
| `03_baseline_cnn.ipynb` | Trains the baseline | ~32 min | 0.583 macro F1 |
| `04_resnet18_transfer_learning.ipynb` | Compares 3 approaches, trains the final model | ~45 min | 0.908 macro F1 |

Run them in order. Notebooks 03 and 04 rebuild the cleaned dataset themselves if they cannot find it, so each works independently in a fresh session — there are no files to move between them.

---

## About exact numbers

Every notebook uses `SEED = 42`, so the five folds come out identical every time and results are directly comparable.

**But GPU training is not perfectly deterministic.** Re-running notebook 03 with no changes at all moved its macro F1 from 0.576 to 0.583. Expect differences of a few thousandths.

If you need exact reproducibility, add this to the configuration block:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

It costs perhaps 10–20% training speed.

The saved results in `outputs/` come from the runs recorded in the notebooks. The numbers in the README match those files.

---

## If something goes wrong

**`manifest.csv not found`** in notebook 03, 04 or 05 — this is normal, not an error. Colab and Kaggle give every notebook its own machine, so one cannot see another's files. The next cell rebuilds the dataset automatically in about five minutes.

**`git clone` fails** in notebook 05 — the notebook prints the actual error. Usually it means no internet access in that session. Fall back to uploading `models/resnet18_config.json` and `models/resnet18_fold2.pt` by hand when prompted.

**`pip install` is very slow** — that is PyTorch, roughly 500 MB. It only happens once.

**`streamlit: command not found`** — the virtual environment is not active. Run `source .venv/bin/activate` again; you will see `(.venv)` appear in your prompt.

**A warning about LibreSSL on macOS** — harmless. macOS ships LibreSSL rather than OpenSSL and `urllib3` mentions it every time. Nothing is broken.

---

## What is deliberately not in this repository

**The dataset images** (~2.1 GB). They are downloaded from Kaggle by the notebooks. Only `data/processed/manifest.csv` is kept, which records exactly which image went into which fold, so the split can be checked without storing the images.

**Four of the five ResNet18 folds** (~45 MB each). Only fold 2 is included, which is enough to make predictions. Re-run notebook 04 to regenerate the rest.

Both are deliberate: the repository holds the code that produces the data, and because the seed is fixed, running that code reproduces it exactly.
