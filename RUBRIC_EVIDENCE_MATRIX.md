# Where to Find the Evidence

This page tells you exactly which file proves each part of the rubric, so nothing has to be hunted for.

**Live demo:** https://capstone-project-fnrnvvgboe6b2i8gczg9ju.streamlit.app

| Criterion | Where to look |
|---|---|
| 1. Problem definition and alignment | `docs/PROJECT_BRIEF.md`, README — Problem Statement, ML Task Type, Success Criteria |
| 2. Data and preprocessing | `notebooks/01_dataset_exploration.ipynb`, `notebooks/02_data_preprocessing.ipynb`, `data/processed/manifest.csv` |
| 3. Modeling and experiments | `notebooks/03_baseline_cnn.ipynb`, `notebooks/04_resnet18_transfer_learning.ipynb`, `docs/EXPERIMENT_LOG.md`, `outputs/resnet18/resnet18_experiments.csv` |
| 4. Evaluation and error analysis | `notebooks/04_resnet18_transfer_learning.ipynb` §7–8, `reports/ERROR_ANALYSIS.md`, `outputs/resnet18/` |
| 5. End-to-end delivery | Live app above, `app.py`, `src/predict.py`, `notebooks/05_inference_demo.ipynb`, `models/` |
| 6. Documentation and reproducibility | `README.md`, `requirements.txt`, `docs/REPRODUCTION_TEST.md` |
| 7. Responsible AI and limitations | `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`, README — Known Limitations |
| 8. Presentation, demo and Q&A | The live app; `notebooks/05_inference_demo.ipynb`; slides submitted separately |

---

## Criterion 1 — Problem definition

| What is required | Where it is |
|---|---|
| A real problem, clearly stated | README — Problem Statement |
| The ML task, its input and its output | README — ML Task Type |
| Measurable success criteria | README — Success Criteria (four targets, all met) |
| Matches the approved brief | `docs/PROJECT_BRIEF.md` |

---

## Criterion 2 — Data and preprocessing

| What is required | Where it is |
|---|---|
| Dataset described: source, size, structure | README — Dataset; notebook 01 §2 |
| Exploration that finds real problems | Notebook 01 — nine sections of checks |
| Cleaning done properly | Notebook 02 §2 — 330 duplicates removed in three passes |
| Feature engineering, or a reason for skipping it | Notebook 02 §4 — a CNN learns its own features from pixels, so none are hand-made |
| Train / validation / test split | Notebook 02 §5 — five stratified folds; notebook 04 §5 uses fold 3 for choosing and folds 0–2 for training |
| No data leakage | Notebook 01 §10–14 and notebook 02 §2 — duplicates found three separate ways |

**The strongest evidence here** is notebook 01 §11–14. The raw download claims 3,067 images, but 330 of them are copies. Left in place, a copy in the test set would have been an image the model had already memorised, and every score after that would have been meaningless.

---

## Criterion 3 — Modeling and experiments

| What is required | Where it is |
|---|---|
| A baseline | Notebook 03 — a CNN built from scratch, 0.583 macro F1 |
| A main model trained by the student | Notebook 04 — ResNet18 fine-tuned, 0.908 macro F1 |
| At least two approaches compared | Notebook 04 §5 — three ways of adapting ResNet18, plus the baseline |
| Several experiments, analysed | `docs/EXPERIMENT_LOG.md` — four runs, each changing one thing |
| Experiments tracked | `outputs/resnet18/resnet18_experiments.csv`, written automatically during the run |
| Final model justified with evidence | Notebook 04 §6, README — Final Model and Justification |

**Worth noticing:** one experiment was included because it was expected to fail. Fine-tuning at a learning rate of 0.001 scored 0.502 — worse than the from-scratch baseline — which shows *why* the chosen setting is right instead of just claiming it.

---

## Criterion 4 — Evaluation and error analysis

| What is required | Where it is |
|---|---|
| Metrics suited to the task, explained | README — Why macro F1 rather than accuracy |
| Results on unseen data | Every result uses 5-fold cross-validation; each image is scored by a model that never trained on it |
| Compared against the baseline | Notebook 04 §8, README — Headline results |
| Error analysis | `reports/ERROR_ANALYSIS.md`, notebook 04 §7 confusion matrix |
| Reliability checks | Notebook 05 §6 — confidence threshold measured, not guessed; notebook 05 §7 — awkward inputs |
| Honest conclusions | README — Known Limitations; `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md` |

---

## Criterion 5 — End-to-end delivery

| What is required | Where it is |
|---|---|
| A working prediction pipeline | `src/predict.py`, `app.py` |
| A reproducible demo | **Live app** (link above), and `notebooks/05_inference_demo.ipynb` |
| Bad input handled safely | `src/predict.py` — `load_and_check()`; notebook 05 §7 tests eight awkward files |
| Model and settings saved and reloaded | `models/resnet18_fold2.pt` with `models/resnet18_config.json` |
| Runs in a clean environment | Notebook 05 §1 clones the repository from scratch; `docs/REPRODUCTION_TEST.md` |
| Sensible on edge cases | Notebook 05 §7 — blank and broken images refused |

**The clean-run proof** is notebook 05 §1. In a fresh Colab session with nothing prepared, it clones this repository, loads the weights, and is ready to classify in about a minute.

---

## Criterion 6 — Documentation and reproducibility

| What is required | Where it is |
|---|---|
| A clear README | `README.md` |
| Setup and run instructions | README — Installation, Running the Project |
| Files organised sensibly | This page; `START_HERE.md` |
| Someone else can reproduce it | `docs/REPRODUCTION_TEST.md` |

Every notebook uses a fixed random seed, so re-running produces the same five folds and comparable results.

---

## Criterion 7 — Responsible AI and limitations

| What is required | Where it is |
|---|---|
| Bias and fairness | `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md` |
| Privacy and safety | Same file — fabric photographs only, no personal data |
| Limitations and misuse | Same file; README — Known Limitations |

**This is backed by demonstrated failures, not claims.** Notebook 05 §5 shows the model answering "horizontal" with 89.2% confidence on plain yellow cotton that has no defect at all.

---

## Criterion 8 — Presentation, demo and Q&A

This criterion is assessed during the defence itself rather than from the repository. The repository provides what the presentation is built on:

| What is required | Where it is |
|---|---|
| A clear presentation | Slides submitted separately |
| A working live demo | The live app (link above); `notebooks/05_inference_demo.ipynb` as a backup |
| Results explained clearly | README — Evaluation Metrics and Results; `reports/ERROR_ANALYSIS.md` |
| Questions answered well | Every claim in this table points at a file, so answers can be shown rather than recalled |

---

## Must-pass requirements

- [x] The project includes a model trained by the student — ResNet18, fine-tuned in notebook 04
- [x] The final model is tested on data it never saw — 5-fold cross-validation over all 2,737 images
- [x] A working end-to-end demo exists — live app, plus notebook 05 and `src/predict.py`
- [x] Reproduction instructions are clear — README and `docs/REPRODUCTION_TEST.md`
- [ ] The student attends the defence
