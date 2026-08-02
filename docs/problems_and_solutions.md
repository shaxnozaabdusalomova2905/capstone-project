# Problems Faced and How They Were Solved

**Project:** Automated Fabric Defect Classification
**Author:** Shakhnoza Abdusalomova

This document records every significant problem encountered while building the project, what caused it, what was done about it, and why that choice was made rather than an alternative.

It is written in the order the problems appeared, from the first notebook to the last.

---

## Summary

| # | Problem | Found in | Solution |
|---|---|---|---|
| 1 | Images displayed in false colours; suspected mislabelling | 01 | Diagnosed as a colour-mode artifact, not bad labels |
| 2 | 309 pre-augmented copies inflating the dataset | 01, 02 | Removed by filename pattern |
| 3 | Duplicate files hiding under different names | 01, 02 | Removed by comparing file contents |
| 4 | Near-identical images from file copying | 01, 02 | Removed by comparing appearance |
| 5 | First duplicate detector flagged 54% of the data | 01 | Diagnosed as hash collisions; used a finer method |
| 6 | No fabric roll or batch information for splitting | 01 | Searched twice, found none, recorded as a limitation |
| 7 | Dataset assembled from several different sources | 01, 02 | Tested whether it leaks the answer; partially safe |
| 8 | Image dimensions predict the class | 02 | Resized to a square instead of padding |
| 9 | Mixed colour formats across classes | 02 | Converted everything to RGB |
| 10 | Severe class imbalance, 62:1 | 02 | Weighted loss, stratified folds, macro F1 |
| 11 | Rare classes too small for a normal train/test split | 02 | 5-fold cross-validation |
| 12 | Baseline CNN confused similar defect types | 03 | Diagnosed; motivated transfer learning |
| 13 | Rare-class scores unstable between folds | 03 | Reported per fold with ranges, not single numbers |
| 14 | Choosing a model on test data would inflate results | 04 | Separate validation fold for model selection |
| 15 | Unclear whether to freeze or fine-tune | 04 | Measured three approaches instead of guessing |
| 16 | Files could not be passed between Colab notebooks | 03, 04, 05 | Notebooks rebuild the dataset themselves |
| 17 | Results and figures went stale after re-running | 03 | Rebuilt output files from the notebook's own record |
| 18 | Demo could not find the model in a clean session | 05 | Weights committed to the repository; repo made public |
| 19 | Review threshold caught almost nothing | 05 | Measured the trade-off and raised it from 0.60 to 0.95 |
| 20 | Model confidently misclassifies blank images | 05 | Added a content check before the model is asked |
| 21 | Model fails on photographs from other sources | 05 | Recorded as a measured limitation |

---

## Notebook 01 — Dataset Exploration

### Problem 1: Images appeared to be in the wrong folders

**What happened.** Displaying one sample image per class showed something alarming. "Broken stitch", "Needle mark" and "Pinched fabric" looked like normal fabric photographs, but "Vertical" and "horizontal" appeared as bright purple and yellow patterns, "hole" was almost entirely black, and "lines" and "stain" had strange colour casts. The first conclusion was that the dataset was mislabelled — that the "hole" images contained no hole.

**What actually caused it.** The images are stored in different colour formats. A colour image holds three numbers per pixel (red, green, blue); a grayscale image holds only one, describing brightness. When a grayscale image is displayed without being told it is grayscale, matplotlib has brightness values but no colours to pair them with, so it applies its default colour scheme — dark values become purple and bright values become yellow.

Looking again at the "Vertical" image with that in mind, the bright yellow streaks *were* the vertical defect. It had been visible all along, just painted in false colours.

**How it was solved.** Section 15 checked the colour format of every file and confirmed the cause: **every** `_processed` file is single-channel grayscale, without exception. The fix in preprocessing is to convert every image to RGB before use.

**Why this mattered.** Recording "the dataset is mislabelled" would have been wrong, and it would have undermined every result that followed. The lesson is to diagnose a surprising observation before acting on it.

---

### Problem 2: The dataset was smaller than it appeared

**What happened.** The download contains 3,067 files. Listing filenames revealed entries such as `10.jpg` sitting beside `10_processed (1).jpg`, `10_processed (2).jpg` and `10_processed (3).jpg`.

**What caused it.** Whoever assembled the dataset had already applied augmentation and saved the results alongside the originals. Three classes were affected:

| Class | Files | Genuinely different images |
|---|---:|---:|
| horizontal | 136 | 34 |
| Vertical | 101 | 32 |
| hole | 281 | 143 |

**Why this is dangerous.** If `10.jpg` is used for training and `10_processed (2).jpg` for testing, the model is being graded on a picture it has already memorised. The reported score would look excellent and mean nothing. This is data leakage, and it is one of the most common ways a machine learning result becomes worthless.

**How it was solved.** All files with `_processed` in the name are excluded during preprocessing.

**Why they were deleted rather than kept as extra training data.** Two reasons. They are altered versions of pictures already in the dataset, so they add no new information. And they exist in only three of the nine classes — so keeping them would let the model learn "grayscale means Vertical, hole or horizontal" instead of learning what those defects look like. Augmentation is instead generated during training and applied evenly to every class.

---

### Problem 3: Duplicate files under unrelated names

**What happened.** Filename checking only catches copies that are *named* like copies. Two identical images saved as `photo_a.jpg` and `photo_b.jpg` would slip through.

**How it was solved.** An MD5 fingerprint was taken of every file's contents. Files sharing a fingerprint are identical regardless of their names. This found **17** duplicates.

**A second thing this checked.** Duplicate groups spanning two different class folders would mean one image carrying two contradictory labels — genuine mislabelling. **Zero** such groups were found, which finally settled the concern raised in Problem 1.

---

### Problem 4: Near-identical images that were not byte-identical

**What happened.** A file that has been opened and re-saved has different bytes but the same picture. MD5 misses these entirely.

**How it was solved.** A perceptual hash compares how images *look* rather than their exact bytes. This found **19 clusters covering 40 images**.

The filenames revealed what they were:

```
exp2_num_249619.png  +  exp2_num_249619 - Copy.png
20180531_135032.jpg  +  20180531_135032(1).jpg
```

Endings like `- Copy` and `(1)` come from copying files on a computer or downloading the same file twice — accidents from assembling the dataset, not separate photographs.

**Result.** Combined with Problems 2 and 3, **330 files** were removed, leaving **2,737** genuinely different images.

---

### Problem 5: The first duplicate detector flagged half the dataset

**What happened.** The first attempt at appearance-based comparison used an 8×8 average hash. It flagged **1,494 images — 54% of the dataset** — as near-duplicates. That is implausible.

**How the error was spotted.** The output was checked rather than trusted. One cluster grouped a broken-stitch image with ten defect-free images. Another paired a studio fabric close-up with a phone photograph from an entirely different collection. Those images are plainly different.

**What caused it.** Shrinking an image to 8×8 leaves only 64 pixels. Fabric close-ups are low-contrast and nearly uniform, so once shrunk they all look alike and many produce the same code by coincidence. These were hash collisions, not duplicates.

**How it was solved.** The method was replaced with a 16×16 difference hash, which compares each pixel against its neighbour rather than against the image's average brightness. This captures edges and texture, which survive shrinking far better, and produces a code four times longer, making accidental matches far less likely.

The count dropped from **54% to 1.5%**, and the surviving clusters were confirmed visually.

**Why this is recorded rather than hidden.** The failed first attempt is kept in the notebook deliberately. It shows a method being tested and corrected rather than trusted blindly, and the second result is more credible because the first was checked.

---

### Problem 6: No way to split the data by production batch

**What happened.** Factory datasets usually contain several photographs of the same fabric roll. If images of one roll are split between training and testing, the model can recognise that roll rather than learning the defect, and the test score is inflated.

**What was tried.** Two attempts to find such grouping:

1. **Filenames** — searched for roll or batch identifiers. None exist.
2. **Appearance** — looked for clusters of visually similar images that might indicate a shooting session. The clusters found turned out to be file-copying accidents (Problem 4), not session structure.

Notably, the coarse hash had appeared to group consecutively numbered files such as `A_02_030` and `A_02_031`, which looked like same-session shots. With the finer method those groupings disappeared — they were collisions too.

**How it was resolved.** It was not solved, because it cannot be. It is recorded openly as a limitation: splitting by production batch is impossible with this dataset, both available methods were tried, and scores may therefore be slightly optimistic.

**Why this matters.** Stating a limitation you cannot fix is stronger than quietly ignoring it. A mentor who spots the issue independently will trust the rest of the work more if it was already declared.

---

### Problem 7: The dataset combines several different collections

**What happened.** Filenames follow completely different conventions by class — `A_02_001.jpg` for some, camera timestamps like `20180531_140300.jpg` for others, plain numbers for others again.

**Why this is dangerous.** If normal images came from one collection and defective images from another, they would differ in camera, lighting and compression. A model could then score highly by detecting *which camera took the photo* rather than whether the fabric is defective. It would appear to work and fail completely in a real factory.

**How it was tested.** Each filename was sorted into a naming-pattern group and the patterns counted per class. The test was whether normal and defective images ever share a source.

**Result — the risk is reduced but not eliminated.** They do share sources:

- The dominant pattern in "defect free" (1,454 files) is also the *only* pattern used by "Broken stitch", "Needle mark" and "Pinched fabric"
- A second pattern appears in "defect free" as well as "stain", "horizontal" and "Vertical"

Because normal and defective images sit inside the same collections, the model cannot separate them by source alone. Recorded as a remaining limitation rather than a solved problem.

---

## Notebook 02 — Data Preprocessing

### Problem 8: Image size gives away the class

**What happened.** Checking image dimensions per class produced an uncomfortable result:

| Class | Dimensions |
|---|---|
| Broken stitch, Needle mark, Pinched fabric | **224×224 — every single image** |
| defect free | mostly 2446×1000 |
| Vertical, horizontal | 640×360 |
| stain | 1984×1488 |

Three classes are already exactly 224×224 and no other class contains a single image at that size. Knowing only an image's dimensions would allow guessing its class correctly a large share of the time.

**Why this nearly caused a serious mistake.** The original plan was **letterbox padding** — shrinking each image to fit and filling the remaining space with black bars, which preserves the original proportions. That is normally the careful choice, because it avoids distorting the picture.

Here it would have leaked the answer. Preserving proportions means the shape of the black bars encodes the original dimensions, and therefore the source collection, and therefore the class. The model could have read the padding instead of the fabric.

**How it was solved.** Every image is resized **directly to a 224×224 square**, accepting that proportions are distorted.

**Why distortion is acceptable here.** The orientation-based classes — "Vertical", "horizontal", "lines" — are defined by *direction*, and squashing an image does not rotate anything. A vertical line stays vertical. Proportion is lost; the feature that identifies the defect is kept.

---

### Problem 9: Colour formats differ by class

**What happened.** The same check showed all 398 "stain" images are stored in grayscale, while most other classes are in full colour. "defect free" is a mixture of colour, grayscale and transparency-carrying images.

**How it was partly solved.** Every image is converted to RGB, which gives the model uniform three-channel input regardless of how the file was stored.

**Why this is only a partial fix.** A grayscale image converted to RGB is still visibly colourless. "Has no colour" therefore remains a usable clue for "stain". This is recorded as a limitation. Notebook 03 weakens it by randomly converting training images to grayscale so the cue becomes unreliable; removing it entirely would mean converting every image to grayscale, which was considered but not done.

---

### Problem 10: Severe class imbalance

**What happened.** After cleaning, the distribution is far worse than the raw file counts suggested:

| Class | Images | Share |
|---|---:|---:|
| defect free | 1,663 | 60.8% |
| Vertical | 27 | 1.0% |

The largest class is **62 times** the size of the smallest. Deduplication made this worse, not better — "defect free" rose from 54.3% to 60.8%.

**Why this is dangerous.** A model can score 60.8% accuracy by always answering "defect free" and never learning anything. Left uncorrected, training pushes it towards exactly that, because ignoring the rare classes barely affects overall error.

**How it was solved — four measures together:**

1. **Weighted loss.** Each class gets a weight based on how rare it is. Getting a "Vertical" image wrong costs 11.26 against 0.18 for "defect free" — a 62× penalty matching the imbalance.
2. **Stratified folds.** Every fold holds the same class proportions, so no fold can end up without a rare class.
3. **Macro F1 as the main metric.** It averages all nine classes equally, so failure on the rare ones cannot hide behind success on the common one.
4. **Per-class recall reported separately**, sorted rarest-first.

**Why weighting rather than oversampling.** Duplicating rare images until every class matches would mean showing the model the same 27 "Vertical" pictures around sixty times each. It would memorise them rather than learn from them. Weighting achieves the same goal without that risk, and variety comes from augmentation applied evenly to every class.

**Evidence it worked.** No class is ignored. In the baseline CNN the correction can even be seen slightly overshooting — "Needle mark" reached 0.954 recall but only 0.560 precision, meaning the model guessed rare classes too freely because missing them was so expensive.

---

### Problem 11: Rare classes too small for a normal split

**What happened.** A conventional 70/15/15 split would give "Vertical" roughly **4 test images**. A recall figure computed from 4 images moves in 25-point jumps depending on a single prediction. That is not a measurement.

**How it was solved.** **5-fold cross-validation.** The data is split into five equal parts and the model trained five times, each time holding out a different part. Every image is tested exactly once, so all 27 "Vertical" images contribute to the score instead of 4.

**The cost.** Five training runs instead of one — about 32 minutes rather than 6. Worth it: the difference between a number that can be reported and one that cannot.

---

## Notebook 03 — Baseline CNN

### Problem 12: The baseline confused similar defect types

**What happened.** A CNN trained from scratch reached only **0.583 macro F1**. The confusion matrix showed the errors were not spread evenly but concentrated in specific pairs:

- **Broken stitch called Needle mark 69 times out of 112**
- **Vertical called horizontal 19 times out of 27**
- "hole" scattered across "lines" and "horizontal", with only 29 of 141 correct

**What caused it.** The confused pairs come from the same source collections and look genuinely similar — Broken stitch and Needle mark are both 224×224 fabric close-ups. A network learning from 2,737 images alone could not find the distinction.

**An unexpected finding.** Class size did not predict difficulty. "Needle mark" (108 images) reached 0.954 recall while "Broken stitch" (112 images) managed 0.250 — nearly identical class sizes, completely different outcomes. What mattered was whether a class resembled another class, not how many examples it had.

**How it was solved.** This is what motivated transfer learning. A network already trained on a million photographs understands edges and texture before it ever sees fabric, so it can make distinctions that a from-scratch model cannot. Notebook 04 raised "Broken stitch" recall from 0.250 to **1.000**.

---

### Problem 13: Rare-class scores were unstable

**What happened.** Overall macro F1 was steady across folds (±0.019), which suggested a reliable result. Breaking it down by class told a different story:

| Class | Recall per fold | Pooled |
|---|---|---|
| horizontal | 0.29, 1.00, 1.00, 0.17, 0.86 | 0.676 |
| Vertical | 0.40, 0.00, 0.00, 0.17, 0.17 | 0.148 |

"horizontal" swings across the entire possible range. "Vertical" found nothing at all in two of the five folds.

**Why the overall figure hid this.** Macro F1 averages nine classes, so two unstable ones are diluted by seven stable ones.

**How it was solved.** Rare-class recall is reported **per fold with its range**, not as a single number. Any figure for "Vertical" or "horizontal" is quoted as approximate.

**Why this cannot be fixed by better modelling.** Each fold holds only 5–7 images of these classes, so one prediction changing moves the recall by around 17 points. The instability comes from having 27 and 34 images, and only more data would remove it.

---

## Notebook 04 — ResNet18

### Problem 14: Choosing a model on test data would inflate the result

**What happened.** Three ways of adapting ResNet18 needed comparing. The obvious approach — train all three, see which scores best on the test data, report that score — is subtly wrong. Picking the setup that happens to look best on the test set means the reported number is partly luck rather than performance.

**How it was solved.** The folds were divided by role:

| Folds | Role |
|---|---|
| 0, 1, 2 | Training |
| 3 | Validation — used to choose the approach |
| 4 | Not involved in the choice at all |

The approach was chosen using fold 3 only. Fold 4's score is reported separately as a clean check: it scored **0.895**, in line with the others, confirming the selection did not distort the result.

**The remaining caveat, stated openly.** Fold 3 also appears in the final cross-validation, so the headline figure is very slightly optimistic. This is written into the notebook rather than left for someone else to notice.

---

### Problem 15: No way to know whether to freeze or fine-tune

**What happened.** A pretrained network can be adapted two ways: lock the existing layers and train only a new final layer, or let everything keep learning. Both have reasonable arguments — freezing avoids overfitting on a small dataset, fine-tuning lets the features adapt to fabric. There is no way to know in advance which wins.

**How it was solved.** All three were measured:

| Approach | Validation macro F1 |
|---|---:|
| Fine-tuning at lr 0.0001 | **0.908** |
| Frozen backbone | 0.711 |
| Fine-tuning at lr 0.001 | 0.502 |

**The third one was included deliberately as a likely failure.** Fine-tuning with too large a learning rate can overwrite what the network learned from ImageNet in the first few batches. It scored **0.502 — worse than the from-scratch baseline CNN**, confirming exactly that.

**Why testing a bad option was worth the time.** It demonstrates *why* the chosen learning rate is right rather than merely asserting it. A negative result is evidence.

---

### Problem 16: Files could not be passed between notebooks

**What happened.** Notebook 02 saved the cleaned dataset, then notebook 03 could not find it. Both Colab and Kaggle give every notebook its own machine, so files written by one are invisible to another.

**What was tried and abandoned.** Saving to Google Drive and copying a zip across. It added a mounting step, an authorisation prompt, and a failure mode — training directly from Drive would also have been very slow, since every image is re-read each epoch.

**How it was solved.** Each notebook **rebuilds the cleaned dataset itself** if it cannot find one, repeating exactly the same steps with the same fixed random seed. The folds come out identical, so results stay comparable.

**Why this is better than transferring files.** The notebooks became independent — any of them can be opened in a fresh session and will work, with no setup, no file transfers and nothing to attach. The rebuild costs about five minutes, which is less than the transfer workflow cost in friction.

---

### Problem 17: Results and figures went stale

**What happened.** Notebook 03 was re-run, producing slightly different numbers — macro F1 moved from 0.576 to 0.583 — because GPU training is not perfectly deterministic. The saved result files and the confusion matrix in the README were still from the earlier run. The notebook and its output files disagreed.

**How it was solved.** The affected files were rebuilt from the notebook's own recorded output, and verified line by line against the printed tables. Two files that could not be recovered — the per-image predictions and the epoch-by-epoch history — were deleted rather than left stale.

**Why deleting was better than keeping.** A missing file is honest; a file that contradicts the notebook beside it is not.

**What would prevent it recurring.** Setting `torch.backends.cudnn.deterministic = True` makes runs reproducible at a small cost in speed. This was identified but not applied retrospectively, since re-running would have produced a third set of numbers and required rewriting every observation again.

---

## Notebook 05 — Prediction Demo

### Problem 18: The demo could not find the model

**What happened.** The demo notebook was designed to run in a clean session by cloning the repository. It failed with "could not find models/resnet18_config.json".

**Two causes, one hidden by the other.** The weights had not been pushed to GitHub yet, and the repository was private so an anonymous clone was refused. The code used `git clone --quiet` and discarded the error message, so the real reason never appeared.

**How it was solved.** Three changes: the clone now captures and prints git's actual error, the repository was made public, and the model weights were committed.

**The size problem this created.** Each ResNet18 fold is about 45 MB, so all five would add over 220 MB to the repository. Only **fold 2** is committed — enough to make predictions — while the rest are regenerated by re-running notebook 04. The baseline CNN's weights are only 1.6 MB each, so all five of those are kept.

**Why the settings file matters as much as the weights.** Saved weights are only a list of numbers. Anything loading them must know which class each output position means, what size images to expect, and which normalisation values were used. Those are saved in `resnet18_config.json` at the same moment as the weights, so they cannot drift apart.

---

### Problem 19: The review threshold caught almost nothing

**What happened.** Predictions below 60% confidence were flagged for human review. Measuring the effect showed it was doing essentially nothing:

| Threshold | Answered automatically | Accuracy on those | Sent to a person |
|---|---|---|---|
| none | 100.0% | 96.9% | 0% |
| 0.60 | 98.7% | 97.0% | 1.3% |
| **0.95** | **90.5%** | **99.8%** | **9.5%** |
| 0.99 | 83.2% | 100.0% | 16.8% |

At 60% it referred 1.3% of images and improved accuracy by one tenth of one percent.

**Why it failed.** Wrong answers averaged **79.4%** confidence while right answers averaged 98.4%. The model genuinely is less sure when wrong — but 79.4% is still a high number, so a 60% bar sits below almost every mistake.

**How it was solved.** Raised to **0.95**, chosen from the table rather than guessed. The effect was immediately visible: the single wrong answer in the sample test (94.7% confidence) went from silently accepted to correctly referred.

**The caveat.** The threshold was chosen using fold 2, and the 99.8% figure comes from that same fold. Strictly it should be selected on one set and reported on another. With a single parameter from a short list the risk is small, but this is recorded as a reasonable choice rather than a guaranteed result.

---

### Problem 20: Blank images are classified confidently

**What happened.** Deliberately awkward inputs were tested. Unreadable files were correctly refused. But a completely black image was classified as **"horizontal" at 94.8% confidence**, and three plain grey images all came back as "horizontal" at around 90%.

**What this revealed.** Shown something with no fabric texture at all, the model does not become uncertain — it has a confident default answer.

**Why this mattered beyond the test.** It explained a separate failure. A photograph of plain yellow cotton had been classified as "horizontal" at 89.2% when it had no defect at all. Smooth featureless fabric looks to the model much like a blank image.

**How it was partly solved.** Raising the threshold to 0.95 flagged the black image — but only by two tenths of a percentage point. A blank image scoring 96% would still pass.

**The real fix.** The command-line tool and the web app both now check the image *before* the model is asked, rejecting anything whose brightness barely varies:

| Input | Brightness variation | Result |
|---|---:|---|
| Solid black | 0.0 | refused |
| Solid grey | 0.0 | refused |
| Real fabric texture | 20.6 | accepted |

**Why a content check rather than a higher threshold.** The threshold responds to how sure the model is, and on inputs unlike anything it has seen the model is confidently wrong. No confidence setting fixes that. The problem has to be caught before the model is consulted.

---

### Problem 21: The model fails on photographs from other sources

**What happened.** Testing on personal phone photographs gave mixed results:

| Image | Prediction | Correct? | Confidence |
|---|---|---|---|
| Grey burlap, torn strip | hole | ✅ | 100.0% |
| White cotton, rust spots | stain | ✅ | 27.4%, flagged |
| Plain yellow cotton, no defect | horizontal | ❌ | 89.2%, not flagged |
| Generated image, damaged fabric | stain | ❌ type | 56.2%, flagged |

**What this shows.** The model detects *that something is wrong* far more reliably than *what is wrong*. On the fourth image the eight defect classes summed to 97.9% against 1.9% for "defect free" — the right answer to the question a factory actually asks, with the wrong defect type attached.

This matches the measured numbers: collapsing the same predictions to a simple defect / no-defect decision gives **98.1% accuracy and 99.8% recall**, against 96.5% for the nine-way task.

**Why this cannot be solved by changing the method.** The model was trained on one dataset's cameras, lighting and fabric types. Different training settings, more epochs, or a different architecture do not create knowledge of fabric the model has never seen. Only more varied data would.

**How it is handled.** Recorded as the project's principal limitation, with the demonstrated failure cases included rather than omitted. The web app states it in its sidebar so anyone using the system sees the caveat alongside the prediction.

---

## What the problems have in common

Looking back over all twenty-one, they fall into three groups.

**Problems that were solved.** Duplicates, colour formats, class imbalance, split strategy, model selection, file transfer between notebooks, blank-image inputs, threshold calibration. These had proper technical answers and were fixed.

**Problems that were diagnosed but not solvable.** Missing batch information, multi-source composition, rare classes with 27 images, generalisation to other cameras. These are properties of the dataset. No amount of modelling effort removes them, and each is documented as a limitation with the evidence behind it.

**Problems caused by my own first attempt.** The 54% false-positive duplicate detection, the letterbox padding that would have leaked the class, the 60% threshold that did nothing, the clone error that hid its own cause. Each was found by checking a result that looked wrong rather than accepting it.

That last group is the one worth noticing. Every one of them would have gone unnoticed if the output had been trusted instead of examined — and each would have quietly damaged the result.
