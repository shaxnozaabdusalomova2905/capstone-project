# Responsible Use and Limitations

What this system can and cannot do, who could be harmed by a wrong answer, and where it should not be used.

Everything here is backed by something measured, not asserted.

---

## In one sentence

This is a **second opinion for a human inspector**, not a replacement for one. It works well on fabric resembling its training data and can be confidently wrong on anything else.

---

## Bias and representativeness

### The training data is very unevenly balanced

| Class | Images | Share |
|---|---:|---:|
| defect free | 1,663 | 60.8% |
| Vertical | 27 | 1.0% |

The largest class is **62 times** the size of the smallest.

Class weighting was used to stop the model ignoring the rare defects, and it worked — "Vertical" recall rose from 0.148 in the baseline to 0.630. But weighting cannot invent information that 27 images do not contain. The rare classes remain the weakest, and their scores swing widely depending on which images are being tested.

**What this means in practice.** The defects the system is worst at are the ones it has seen least. If a factory's fabric happens to fail mostly in those ways, the system will be less useful there than these numbers suggest.

### The data comes from a small number of collections

Exploration found that image size and colour format differ systematically by class. Three classes are stored at exactly 224×224 and no other class contains a single image at that size. All 398 "stain" images are stored in grayscale while most other classes are in colour.

This creates a risk that the model identifies *where an image came from* rather than what is wrong with the fabric.

**Why we believe it is not doing that, mostly.** Normal and defective images share sources, so the model cannot separate defect from non-defect by source alone. More convincingly, "Broken stitch", "Needle mark" and "Pinched fabric" all come from the *same* collection at the same size and format — nothing about the file distinguishes them — and all three are classified perfectly. That has to be genuine visual learning.

**What we cannot rule out.** Source cues may still help with the other classes. It is recorded as an open limitation.

### It has seen one kind of fabric

All training images come from one public dataset, photographed with particular cameras under particular lighting. The system has never seen most fabric types, colours or weaves that exist.

**This was tested and it failed.** A photograph of plain yellow cotton with no defect was classified as "horizontal" with **89.2% confidence** — and the confidence score gave no warning at all.

---

## Privacy and safety

### Privacy

The dataset contains photographs of fabric only. There are no people, faces, names, or personal information of any kind. The dataset is released under **CC0 — public domain**.

The web app does not store uploaded images. They are held in memory long enough to make a prediction and then discarded.

**One thing a deployer should consider.** In a real factory, images of a production line could reveal commercially sensitive information — what is being made, at what volume, with what defect rates. That is a business confidentiality question rather than a personal privacy one, but it would need addressing before deployment.

### The main safety risk is misplaced trust

A system reporting 96.5% accuracy invites people to stop checking its work.

Three things make that dangerous here:

1. **It is worst at the rare defects** — which are also the ones a human inspector is least practised at spotting. That is exactly where an automated second opinion feels most valuable and is most likely to be trusted without question.
2. **Confidence does not indicate correctness on unfamiliar input.** Testing with blank images showed the model answering "horizontal" at around 90% confidence when given something with no fabric texture at all. It does not become uncertain outside its experience — it becomes confidently wrong.
3. **It misses roughly 1 defect in 500** on this dataset, and an unknown proportion on fabric it has not seen.

### What was done to reduce that risk

| Measure | Where |
|---|---|
| Predictions below 95% confidence are flagged for a person | `app.py`, `src/predict.py` |
| The threshold was chosen by measurement, not guessed | Notebook 05 §6 |
| Blank or near-blank images are refused before the model is asked | `src/predict.py` — `load_and_check()` |
| Unreadable files are refused with a clear reason, never guessed at | Notebook 05 §7 |
| The app states its limitations on screen, beside every prediction | `app.py` sidebar |

That last one matters. The caveats are shown to whoever is using the system, not buried in a document they will never read.

---

## Known limitations

### Two classes have too few images

"Vertical" (27 images) and "horizontal" (34) are the weakest classes and are mostly confused with each other. Their recall swings between 0.33 and 1.00 depending on which fold is being tested.

Any single figure for these two classes overstates what is actually known. They should always be quoted with their range.

**This cannot be fixed by better modelling.** It needs more images.

### Generalisation is untested

Every number in this project comes from one dataset. Performance on fabric photographed with different equipment, under different lighting, is unknown — and one informal test suggests it is poor.

**This is the most important limitation and the least quantified.** The honest position is: *strong on this dataset, untested elsewhere, with one demonstrated failure.*

### No production batch information

Ideally, images of the same fabric roll would be kept together when splitting the data, so the model cannot recognise a roll it has already seen. The dataset records no roll or batch identifiers.

Two attempts were made to recover that grouping — from filenames, and by finding visually similar images — and neither found any. Scores may therefore be slightly optimistic.

### Colour is still a partial clue

All "stain" images are stored in grayscale while most other classes are in colour. Converting everything to RGB gives the model uniform input, but a grayscale image still looks colourless, so "has no colour" remains usable as a hint.

Random grayscale conversion during training weakens this. It does not remove it.

### Only one model's weights are published

The repository contains one of the five trained folds, to keep its size reasonable. Averaging all five would be slightly more accurate.

---

## Where this should and should not be used

### Appropriate

- As a **second opinion** alongside a human inspector
- To **flag rolls** that deserve a closer look
- To **gather statistics** on which defect types occur most often
- As a **teaching aid** for new inspectors learning to recognise defect types

### Not appropriate

- **As the only thing deciding whether fabric ships.** It misses roughly 1 defect in 500 here, and an unknown number on unfamiliar fabric.
- **For safety-critical textiles** — medical fabric, protective equipment, anything where a defect could cause injury.
- **On fabric visibly unlike the training data** — different weaves, colours or lighting. It will still answer confidently, and may be wrong.
- **As evidence in a commercial dispute** about quality between a supplier and a customer.
- **To assess or monitor inspector performance.** It is not accurate enough to judge a person's work, and using it that way would be unfair to them.

---

## Transparency

Every result in this project comes from 5-fold cross-validation on images the model never trained on.

The failures are reported alongside the successes:

- The experiment that performed worse than the baseline is included, not hidden
- The yellow-fabric failure is documented with its confidence score
- Unstable class scores are quoted with their range rather than a single figure
- The threshold was raised from 0.60 to 0.95 because measurement showed the original setting caught almost nothing, and that change is recorded

Where a number is uncertain, it is described as uncertain.
