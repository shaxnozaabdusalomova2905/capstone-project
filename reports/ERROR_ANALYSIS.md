# Error Analysis

Where the model goes wrong, and why.

All numbers come from 5-fold cross-validation over all 2,737 images. Every image was judged by a model that had never seen it during training.

**Source:** `notebooks/04_resnet18_transfer_learning.ipynb` §7, `outputs/resnet18/resnet18_predictions.csv`

---

## The short version

The model is right **96.5%** of the time, with a macro F1 of **0.908**.

The mistakes it does make are not spread evenly. They fall into a small number of specific pairs, and those pairs are defects that genuinely look alike.

![ResNet18 confusion matrix](../assets/resnet18_confusion_matrix.png)

---

## What it gets right

| Class | Images | Recall |
|---|---:|---:|
| Broken stitch | 112 | **1.000** |
| Needle mark | 108 | **1.000** |
| Pinched fabric | 108 | **1.000** |
| stain | 398 | 0.995 |
| defect free | 1,663 | 0.970 |
| hole | 141 | 0.922 |
| lines | 146 | 0.897 |
| horizontal | 34 | 0.794 |
| Vertical | 27 | 0.630 |

Three classes are perfect — not one mistake across all five folds.

That is worth pausing on. In the baseline CNN, "Broken stitch" was called "Needle mark" 69 times out of 112. These three classes come from the same source collection and are all 224×224 close-ups of fabric, so nothing about the file itself distinguishes them. The model had to learn the visual difference, and with transfer learning it did.

---

## What it gets wrong

Every remaining error is in one of three groups.

### 1. Vertical mistaken for horizontal

**8 of 27 "Vertical" images**, plus 4 "horizontal" images called "Vertical".

These are the two smallest classes — 27 and 34 images between them. Both are thin line-shaped marks, and after resizing to 224×224 the difference is which way the line runs.

We deliberately avoided 90° rotations during augmentation for exactly this reason: turning an image a quarter turn would have changed a vertical defect into a horizontal one while the label stayed put, creating deliberately wrong training data. That decision was correct but not sufficient — the classes are simply too similar, and there is not enough data to separate them reliably.

**Can this be fixed?** Not by better modelling. With 27 images, one prediction changing moves that class's recall by about 17 points. Only more images would help.

### 2. hole and lines confused with each other

**13 "lines" images called "hole"**, and **9 "hole" images called "lines"**.

Both are dark marks on fabric. A long thin hole and a dark line look much the same at this resolution.

### 3. stain and defect free

**49 clean images flagged as "stain"**, and **2 stains called clean**.

The imbalance here is deliberate. The class weighting used during training penalises missing a defect far more than raising a false alarm — 11.26 against 0.18 for "defect free", a 62× difference. The model has been pushed to err on the side of caution.

---

## The numbers that matter to a factory

| | Count | Rate |
|---|---:|---:|
| Defects passed off as normal | **2 of 1,074** | **0.2%** |
| Clean fabric wrongly flagged | 50 of 1,663 | 3.0% |

These are not equally bad. A missed defect ships to the customer and may come back as a returned order. A false alarm costs an inspector one extra look at a good roll.

The model is roughly **15 times more likely to raise a false alarm than to miss a defect**, which is the right way round.

**For comparison**, the baseline CNN missed 10 defects and raised 157 false alarms. ResNet18 improved on both at once — usually catching more defects means more false alarms, and the two trade against each other. Here the model simply got better at the task.

---

## Rarity is not what makes a class hard

The obvious explanation for the errors would be "the rare classes are hard because they are rare". The data does not support that.

| Class | Images | Recall |
|---|---:|---:|
| Pinched fabric | 108 | 1.000 |
| Needle mark | 108 | 1.000 |
| Broken stitch | 112 | 1.000 |
| horizontal | 34 | 0.794 |
| Vertical | 27 | 0.630 |

Three classes with roughly 110 images each are perfect. Meanwhile "hole", with 141 images — *more* than any of them — sits at 0.922.

What actually predicts difficulty is **whether a class resembles another class**. "Vertical" and "horizontal" are hard because they look like each other, and being small makes it worse. "Broken stitch" is easy despite having 112 images because, once the model can see texture properly, it looks different from everything else.

This was even clearer in the baseline CNN, where "Needle mark" (108 images) reached 0.954 recall while "Broken stitch" (112 images) managed 0.250 — nearly identical class sizes, completely different outcomes.

---

## How stable are these numbers?

Overall macro F1 varies by only **±0.016** across the five folds, which is reassuring.

The rare classes are a different story:

| Class | Recall in each fold | Pooled |
|---|---|---:|
| Vertical | 0.60, 0.60, 0.80, 0.83, 0.33 | 0.630 |
| horizontal | 1.00, 0.57, 0.57, 0.83, 1.00 | 0.794 |
| Broken stitch | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 |
| Needle mark | 1.00, 1.00, 1.00, 1.00, 1.00 | 1.000 |
| hole | 0.86, 0.89, 0.96, 0.97, 0.93 | 0.922 |

"Vertical" ranges from 0.33 to 0.83. Each fold holds only 5 or 6 of its images, so one prediction changing moves the number a long way.

**This is why "Vertical" and "horizontal" should always be quoted with their range**, never as a single figure. A bare "0.630" claims more precision than the data supports.

---

## Failures outside the dataset

The numbers above describe images from the same dataset. Testing on photographs taken on a phone gives a different picture.

| Image | Prediction | Correct? | Confidence |
|---|---|---|---|
| Grey burlap with a torn strip | hole | ✅ | 100.0% |
| White cotton with rust spots | stain | ✅ | 27.4%, flagged for review |
| Plain yellow cotton, no defect | horizontal | ❌ | 89.2%, **not flagged** |

The third row is the important one. The model was confidently wrong on clean fabric, and the confidence score gave no warning.

**Why it happened.** Testing with deliberately blank images (`notebooks/05_inference_demo.ipynb` §7) showed that when the model is given something with no fabric texture at all, it does not become uncertain — it confidently answers "horizontal", at around 90%. Smooth plain cotton looks much the same to it as a blank image.

**What was done about it.** Two changes. The review threshold was raised from 0.60 to 0.95 after measuring that the lower setting caught almost nothing. And a check was added that examines the image *before* the model is asked, rejecting pictures whose brightness barely varies — because no confidence setting can catch a model that is confidently wrong.

**What remains.** The model has only ever seen one dataset's cameras, lighting and fabric types. On anything else, performance is unknown. This is recorded in `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md`.

---

## Where the model would improve most

1. **More images of "Vertical" and "horizontal".** With 27 and 34, these two account for most of the remaining error and no modelling change will fix them.
2. **A measured test on other cameras.** Photographing and labelling 30–50 fabric samples would turn "it fails on unfamiliar fabric" into an actual number.
3. **Stronger colour augmentation.** The yellow-fabric failure suggests the model leans on colour more than it should. Pushing the colour jitter harder is one experiment away from being answered.
