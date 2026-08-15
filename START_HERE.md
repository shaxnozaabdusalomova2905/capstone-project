# Start Here

A suggested route through this project, for anyone reviewing it.

**The fastest way to see it working:** open the live app and classify a fabric image.
👉 https://capstone-project-fnrnvvgboe6b2i8gczg9ju.streamlit.app

---

## Reviewer route

| Step | What to look at | Why |
|---|---|---|
| 1 | `README.md` | The problem, the results, how to run it |
| 2 | `RUBRIC_EVIDENCE_MATRIX.md` | Which file proves each rubric criterion |
| 3 | `docs/PROJECT_BRIEF.md` | The agreed scope |
| 4 | `notebooks/01_dataset_exploration.ipynb` | What was wrong with the data, and how it was found |
| 5 | `notebooks/02_data_preprocessing.ipynb` | Cleaning, splitting, handling the imbalance |
| 6 | `notebooks/03_baseline_cnn.ipynb` | The baseline that makes the final result meaningful |
| 7 | `notebooks/04_resnet18_transfer_learning.ipynb` | Three approaches compared; the final model |
| 8 | `reports/ERROR_ANALYSIS.md` | Where the model goes wrong, and why |
| 9 | `notebooks/05_inference_demo.ipynb` | The model working on unseen images |
| 10 | `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md` | Where it should not be used |

---

## If you only have five minutes

1. Open the **live app** and classify an image.
2. Read the **Headline results** section of the README.
3. Look at `assets/resnet18_confusion_matrix.png`.

That covers what was built, how well it works, and where it fails.

---

## The three things most worth a closer look

**The duplicate hunt** — `notebooks/01_dataset_exploration.ipynb` §11–14.
The download claims 3,067 images. Only 2,737 are genuinely different. They were found three separate ways, because each method catches something the others miss. Left in place, a copy in the test set would have been an image the model had already memorised.

**The experiment that was expected to fail** — `notebooks/04_resnet18_transfer_learning.ipynb` §5.
Three ways of adapting ResNet18 were compared, including one predicted to damage the pretrained network. It scored 0.502 — worse than the from-scratch baseline. Including it shows why the chosen setting is right rather than merely asserting it.

**The failure case** — `notebooks/05_inference_demo.ipynb` §5.
Plain yellow cotton with no defect at all, classified as "horizontal" with 89.2% confidence and not flagged for review. Testing with blank images explained why: given nothing to look at, the model does not hesitate — it confidently answers "horizontal".

---

## What is where

```
├── README.md                        the main document
├── RUBRIC_EVIDENCE_MATRIX.md        where each criterion is proved
├── START_HERE.md                    this page
│
├── notebooks/                       the five pipeline stages, in order
├── src/predict.py                   command-line classifier
├── app.py                           the web app
│
├── models/                          trained weights + settings to reload them
├── outputs/                         metrics and figures, one folder per model
├── assets/                          figures used in the README, and demo images
├── data/processed/                  which image went into which fold
│
├── docs/
│   ├── PROJECT_BRIEF.md             the agreed scope
│   ├── EXPERIMENT_LOG.md            every run, what changed, what was learned
│   ├── REPRODUCTION_TEST.md         how to verify it yourself
│   ├── RESPONSIBLE_AI_AND_LIMITATIONS.md
│   └── problems_and_solutions.md    21 problems hit during the work
│
└── reports/ERROR_ANALYSIS.md        where the model goes wrong
```

---

## Results at a glance

| | Baseline CNN | ResNet18 |
|---|---:|---:|
| Macro F1 | 0.583 | **0.908** |
| Accuracy | 0.837 | 0.965 |
| Defects missed | 10 of 1,074 | **2 of 1,074** |

From 5-fold cross-validation over 2,737 images. Every image was judged by a model that had never seen it.

**The honest caveat:** these numbers describe this dataset. On fabric photographed with different equipment the model can be confidently wrong, and one demonstrated case is documented.
