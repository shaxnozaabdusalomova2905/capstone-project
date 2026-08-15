# Project Brief

**Author:** Shakhnoza Abdusalomova
**Track:** Individual Project Track

---

## Project title

Automated Fabric Defect Classification for Textile Quality Inspection Using Deep Learning

---

## Project track

**Track 1 — Individual Project Track.** The idea is my own rather than one of the ten prepared scenarios.

---

## Problem statement

In textile factories, fabric is inspected by hand. An inspector watches the material pass by and marks anything that looks wrong.

There are three difficulties with this. It is repetitive work, so attention drops over a long shift. Different inspectors disagree about the same piece of fabric. And the cost of the two possible mistakes is very uneven — a defect that gets missed reaches the customer and may cause a returned order, while a false alarm only costs somebody a second look at a good roll.

This project builds a system that looks at a photograph of fabric and says whether it has a defect, and if so what kind. It is meant to support an inspector, not replace one.

---

## Who it is for

**The inspector on the production line.** They get a second opinion on each piece of fabric, and a prompt to look more closely when the system is unsure.

**The quality manager.** They get a record of which defect types appear most often, and fewer faulty rolls reaching customers.

---

## The machine learning task

**Multi-class image classification** — one photograph in, one label out.

| | |
|---|---|
| Input | A photograph of fabric, resized to 224×224 pixels in colour |
| Output | One of nine labels, plus a confidence score |
| Labels | 8 defect types + 1 "no defect" class |
| Defect types | hole, stain, lines, horizontal, Vertical, Broken stitch, Needle mark, Pinched fabric |

Because one of the nine classes is "defect free", the system answers both questions a factory asks: *is this fabric faulty?* and *what is wrong with it?*

---

## Dataset plan

**Source:** Multi-Class Fabric Defect Detection Dataset on Kaggle
(`ziya07/multi-class-fabric-defect-detection-dataset`, version 3)

**Licence:** CC0 — public domain, free to use

**Size:** about 2.1 GB, 3,067 image files in nine folders, one folder per class

**How it is used.** The images are not stored in this repository — they are downloaded automatically by the notebooks. Only a small file recording which image went into which fold is kept, so the exact split can be checked.

**A problem found during exploration.** The download contains 3,067 files but only **2,737 genuinely different images**. The rest are copies. These are removed before any model is trained, because a copy appearing in both training and testing would make the results meaningless.

---

## Success criteria

| Goal | Target |
|---|---|
| Beat a simple network trained from scratch | Higher macro F1 than the baseline |
| Find every defect type at a useful rate | Recall above 0.5 for all nine classes |
| Rarely let a defect through | Fewer than 2% of defects labelled "defect free" |
| Anyone can reproduce the results | Fixed random seed, written instructions |

**Why macro F1 rather than accuracy.** After cleaning, 60.8% of the images are "defect free" and the rarest class has only 27. A model could score 60.8% accuracy by always answering "defect free" and learning nothing at all. Macro F1 treats all nine classes as equally important, so failing on the rare ones cannot be hidden.

Recall for each class is also reported separately, because in a factory a missed defect is the expensive mistake.

---

## Scope

**What is included:**

- Exploring the dataset and finding its problems
- Cleaning: removing copies, making colours and sizes consistent
- A simple network built from scratch, as something to beat
- ResNet18 adapted using transfer learning, as the main model
- Comparing several ways of adapting it, and explaining the choice
- Testing with cross-validation so every image is judged fairly
- Error analysis: which defects get confused with which
- A working demo anyone can open in a browser

**What is not included:**

- Marking *where* on the fabric the defect is (this needs box-by-box labelling the dataset does not have)
- Live video from a factory camera
- Rating how serious a defect is
- Connecting to real factory equipment

---

## The final demo

A web page where anyone uploads a fabric photograph and immediately sees the predicted defect type, how confident the model is, and whether a person should double-check it.

It runs online with nothing to install, and can also be run locally or from a notebook.

---

## Changes from the original plan

**EfficientNet-B0 was dropped.** The first plan compared ResNet18 with EfficientNet-B0. After mentor feedback the comparison became: a simple network from scratch → ResNet18. This answers a more useful question — *does transfer learning help?* — instead of just picking between two pretrained networks.

**Cross-validation replaced a single split.** The rarest class has 27 images. A normal test split would have given it about 4 test images, and a score from 4 images swings wildly on a single prediction. Five-fold cross-validation lets every image be tested once, which makes the rare-class numbers reportable.

**Squashing images replaced padding.** The original plan was to pad images to keep their proportions. Exploration showed that image size gives away the class — three classes are stored at exactly 224×224 and no other class has a single image that size. Padding would have preserved that clue and let the model read the padding instead of the fabric.

---
