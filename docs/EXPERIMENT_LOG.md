# Experiment Log

Every training run, what was changed, and what it taught us.

The rule throughout: **change one thing at a time**. Every run below uses the same 2,737 images, the same five folds, the same augmentation, the same class weights and the same random seed. Only the named change differs, so any difference in the result can be attributed to it.

**Raw record:** `outputs/resnet18/resnet18_experiments.csv` (written automatically during the run)

---

## The runs

| # | Approach | What changed | Epochs | Macro F1 | Where |
|---|---|---|---:|---:|---|
| 1 | Simple CNN, from scratch | starting point | 25 | 0.583 | Notebook 03 |
| 2 | ResNet18, frozen | only the final layer learns | 15 | 0.711 † | Notebook 04 §5 |
| 3 | **ResNet18, fine-tuned, lr 0.0001** | **all layers learn, small steps** | 15 | **0.908** † | Notebook 04 §5 |
| 4 | ResNet18, fine-tuned, lr 0.001 | all layers learn, large steps | 15 | 0.502 † | Notebook 04 §5 |
| 5 | **Run 3 repeated on all folds** | full cross-validation | 25 | **0.908** | Notebook 04 §6 |

† validation score, measured on fold 3 while choosing between approaches

---

## Run 1 — Simple CNN, built from scratch

**What it was.** Four convolution blocks, 391,689 parameters, trained on the 2,737 images and nothing else.

**Why it exists.** Without it, "0.908" is just a number. This shows what the dataset gives you with no outside knowledge, so the gap between run 1 and run 5 is what transfer learning actually contributed.

**Result.** 0.583 macro F1, 0.837 accuracy.

**What it taught us.** The errors were not random. "Broken stitch" was called "Needle mark" 69 times out of 112, and "Vertical" was called "horizontal" 19 times out of 27. The model could tell *something* was wrong but could not distinguish similar-looking defects.

It also showed that class size does not predict difficulty: "Needle mark" (108 images) reached 0.954 recall while "Broken stitch" (112 images) managed 0.250.

**Learning curves.** Held-out performance stopped improving around epoch 5 and then oscillated. Training longer would not have helped — the limit was the architecture and the amount of data.

---

## Run 2 — ResNet18 with a frozen backbone

**What changed.** Started from ResNet18 already trained on ImageNet. Locked every layer and trained only a new nine-class output layer — **4,617 trainable parameters out of 11,181,129**, or 0.04% of the network.

**Why try it.** With only 2,737 images, letting 11 million parameters loose risks the model memorising the data. Freezing avoids that entirely and trains in under three minutes.

**Result.** 0.711 validation macro F1.

**What it taught us.** ImageNet's ready-made features are genuinely useful on fabric — 0.711 already beats the from-scratch baseline by a wide margin, while training almost nothing. But the features are generic, and they were never adapted to fabric texture specifically.

Its training loss never fell below 0.61, and its validation score levelled off around 0.68–0.71. With only the last layer able to change, there is a ceiling on how well it can fit.

---

## Run 3 — ResNet18 fine-tuned at learning rate 0.0001

**What changed.** Same starting point, but every layer allowed to keep learning, at a deliberately small learning rate.

**Why try it.** Small steps refine what the network already knows rather than overwriting it, letting the features adapt to fabric while keeping the ImageNet knowledge intact.

**Result.** **0.908** validation macro F1 — the best of the three, by 0.197 over the frozen version.

**What it taught us.** Adapting the features is worth far more than using them as they are. The gap between run 2 and run 3 is the single largest improvement in the project.

Its training loss dropped from 1.17 to 0.18 and its validation score reached 0.91 by epoch 10, then held steady — it learned quickly and then had little left to fix.

---

## Run 4 — ResNet18 fine-tuned at learning rate 0.001

**What changed.** The same as run 3, but with steps ten times larger.

**Why try it — this run was expected to fail.** Fine-tuning a pretrained network with too large a learning rate can overwrite what it learned in the first few batches, leaving a damaged network and no time to relearn.

**Result.** **0.502** — *worse than the from-scratch baseline in run 1.*

**What it taught us.** Exactly what was predicted. Its training loss stayed high (0.87) and its validation score wandered between 0.46 and 0.54 with no upward trend — the signature of a model whose useful starting point has been destroyed, now effectively learning from scratch with a network far too large for the data.

**Why a failed run is worth the time.** It demonstrates *why* the chosen learning rate is right rather than merely asserting it. Reporting only the setting that worked would leave "why 0.0001?" unanswered.

---

## Run 5 — The chosen approach, across all five folds

**What changed.** Run 3's settings, trained five times, each time holding out a different fold, for the full 25 epochs to match the baseline.

**Result.**

| Fold | Macro F1 | Accuracy |
|---|---:|---:|
| 0 | 0.918 | 0.967 |
| 1 | 0.900 | 0.954 |
| 2 | 0.894 | 0.971 |
| 3 | 0.930 | 0.969 |
| 4 | 0.895 | 0.965 |

**Pooled across all 2,737 predictions: 0.908 macro F1, 0.965 accuracy.**

**What it taught us.** The folds agree closely — ±0.016 — so the headline number describes the model rather than the luck of the split.

Fold 4 is the important check. It played no part in choosing the approach, and scored 0.895, in line with the others. That tells us selecting on fold 3 did not distort the result.

Three classes — Broken stitch, Needle mark and Pinched fabric — are perfect in every fold.

---

## How the approach was chosen

The three ResNet18 runs were compared on **fold 3 only**, with folds 0–2 for training and fold 4 untouched. Choosing on the same data used to report the final score would have inflated it.

| Folds | Role during selection |
|---|---|
| 0, 1, 2 | Training |
| 3 | Validation — used to choose |
| 4 | Not involved at all |

**Selected:** run 3 — ResNet18, all layers fine-tuned, learning rate 0.0001, 25 epochs.

**Why:** highest validation score by a wide margin, stable across folds, and confirmed by fold 4 which had no part in the decision.

**Trade-off accepted:** fold 3 does appear in the final cross-validation, so the headline figure is very slightly optimistic. Strictly, the fold used for choosing should be excluded from the reported score entirely.

---

## Two more settings that were measured, not guessed

### The review threshold — 0.60 raised to 0.95

The demo flags low-confidence predictions for a human. The original setting of 0.60 was chosen arbitrarily; measuring it showed it was doing almost nothing.

| Threshold | Answered automatically | Accuracy on those | Sent to a person |
|---|---|---|---|
| none | 100.0% | 96.9% | 0% |
| 0.60 | 98.7% | 97.0% | 1.3% |
| **0.95** | **90.5%** | **99.8%** | **9.5%** |
| 0.99 | 83.2% | 100.0% | 16.8% |

At 0.60 it referred 1.3% of images and improved accuracy by one tenth of one percent. At 0.95 it handles 90.5% alone at 99.8% accuracy.

The effect was immediately visible: a wrong answer with 94.7% confidence went from silently accepted to correctly referred.

**Evidence:** notebook 05 §6.

### Number of epochs — 25

Both models plateau well before 25 epochs — the baseline around epoch 5, ResNet18 around epoch 10. 25 was used for both so neither was cut short and the comparison stays fair.

**Evidence:** learning curves in notebooks 03 §5 and 04 §6.

---

## Reproducibility

Every run uses `SEED = 42`, so the five folds come out identical each time.

One caveat: GPU training is not perfectly deterministic. Re-running notebook 03 with no changes moved its macro F1 from 0.576 to 0.583. Setting `torch.backends.cudnn.deterministic = True` would remove this at a small cost in speed.
