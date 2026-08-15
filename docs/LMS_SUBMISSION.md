# Capstone Project Submission

**Full name:** Shakhnoza Abdusalomova

**Project track:** Individual Project Track

**Project title:** Automated Fabric Defect Classification for Textile Quality Inspection Using Deep Learning

**Repository URL:** https://github.com/shaxnozaabdusalomova2905/capstone-project

**Live demo:** https://capstone-project-fnrnvvgboe6b2i8gczg9ju.streamlit.app

**Access instructions:** The repository is public and the demo needs no login. Nothing has to be installed to try it — open the demo link, pick a sample image or upload your own, and the model classifies it in the browser.

---

## Short project description

In textile factories, fabric is checked by hand. The work is repetitive, inspectors disagree with each other, and the two possible mistakes cost very different amounts: a missed defect reaches the customer, while a false alarm only costs one extra look.

This project trains a model to classify a photograph of fabric into eight defect types plus a "no defect" class, so an inspector gets an automatic second opinion.

### The data, and what was wrong with it

The dataset is the Multi-Class Fabric Defect Detection Dataset from Kaggle (CC0, public domain) — 3,067 images across nine classes.

Exploration found that only **2,737 of those images are genuinely different**. The rest are copies, found three separate ways: by filename, by comparing file contents, and by comparing how the images look. All were removed before any model was trained. Leaving them in would have meant testing the model on pictures it had already memorised, making every later number meaningless.

The classes are also very uneven — 60.8% of the images are "defect free" and the rarest class has only 27. So accuracy is not used as the main measure: a model could score 60.8% by always answering "defect free" and learning nothing. **Macro F1** is used instead, because it treats all nine classes as equally important, and recall is reported for each class separately.

### What was built

| Model | Result |
|---|---:|
| A simple network built from scratch, as a reference | 0.583 macro F1 |
| ResNet18, only the last layer trained | 0.711 |
| **ResNet18, fully fine-tuned (final model)** | **0.908** |
| ResNet18, fine-tuned with too large a step | 0.502 |

The last row was included on purpose, because it was expected to fail — taking steps that large destroys what the network already learned, and it scored worse than the from-scratch baseline. Including it shows why the chosen setting is right rather than just claiming it.

Everything was held constant between the models: the same images, the same five folds, the same augmentation, the same class weights, the same random seed. The only difference is the model itself.

### Results

Every number comes from five-fold cross-validation, so each image was judged by a model that had never seen it.

| Measure | Baseline | Final model |
|---|---:|---:|
| Macro F1 | 0.583 | **0.908** |
| Accuracy | 0.837 | 0.965 |
| Defects wrongly passed as normal | 10 of 1,074 | **2 of 1,074 (0.2%)** |
| Clean fabric wrongly flagged | 157 of 1,663 | 50 of 1,663 (3.0%) |

Every class improved. Three of them — Broken stitch, Needle mark and Pinched fabric — are now classified perfectly in every fold. The largest gain was Broken stitch, from 0.250 recall to 1.000.

### What it cannot do

Two classes, "Vertical" and "horizontal", have only 27 and 34 images. Their scores swing widely between folds and should be read as approximate rather than exact.

More importantly, the model was trained on one dataset's cameras, lighting and fabric types. Tested on my own phone photographs it can be **confidently wrong** — one photo of plain yellow cotton with no defect at all was classified as "horizontal" with 89.2% confidence. This is written up as the project's main limitation, with the failure case included rather than hidden.

The system is a second opinion for an inspector, not a replacement.

---

## Where to find things in the repository

| | |
|---|---|
| Start here | `START_HERE.md` |
| Which file proves which criterion | `RUBRIC_EVIDENCE_MATRIX.md` |
| Project brief | `docs/PROJECT_BRIEF.md` |
| The five notebooks | `notebooks/` |
| Trained model and its settings | `models/` |
| Results and figures | `outputs/`, `assets/` |
| Error analysis | `reports/ERROR_ANALYSIS.md` |
| Limitations and responsible use | `docs/RESPONSIBLE_AI_AND_LIMITATIONS.md` |
| How to reproduce it | `docs/REPRODUCTION_TEST.md` |

## How to run the demo

**Easiest:** open the live demo link above. Nothing to install.

**In Colab:** open `notebooks/05_inference_demo.ipynb` and run it. It clones the repository, loads the saved model, and classifies images in about a minute. No training and no dataset download needed.

**On your own machine:**

```bash
git clone https://github.com/shaxnozaabdusalomova2905/capstone-project.git
cd capstone-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
