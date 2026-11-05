# Multi-Modality Acquisition and Site Shift Stress Test

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Outputs Part 1](https://zenodo.org/badge/DOI/10.5281/zenodo.23967820.svg)](https://doi.org/10.5281/zenodo.23967820)
[![Outputs Part 2](https://zenodo.org/badge/DOI/10.5281/zenodo.23967824.svg)](https://doi.org/10.5281/zenodo.23967824)
[![Outputs Part 3](https://zenodo.org/badge/DOI/10.5281/zenodo.23967826.svg)](https://doi.org/10.5281/zenodo.23967826)

Code of the paper **"Clean Performance as a Predictor of Robustness in Medical Image Segmentation: A Pre-Specified
Study of Acquisition and Site Shift"**. The link to the paper will be available on publication.

## Overview

Segmentation models are chosen by their Dice score on clean test images, on the assumption that the best model today
remains the best when the scanner, the acquisition protocol or the hospital changes. This project tests that
assumption end to end. It trains a balanced model zoo on five imaging modalities, degrades every test image with
physically modelled degradations at eight severities of **equal structural similarity (SSIM 0.95 to 0.60)**, evaluates
every network on clean, degraded and external-site images, and runs a pre-specified statistical analysis: whether
clean Dice predicts degraded Dice, what model selection costs, what drives robustness, and how far a chosen model can
be trusted.

| | |
|---|---|
| **Tasks** | ISIC 2016 (dermoscopy), Kvasir-SEG (endoscopy), TN3K (thyroid ultrasound), REFUGE2 (fundus, neuroretinal rim), Darwin (chest X-ray, lung fields) |
| **External sites** | PH2, CVC-ColonDB, DDTI, Drishti-GS1, Montgomery + Shenzhen |
| **Models** | U-Net, Attention U-Net, DeepLabv3+, HRNet-W32, TransUNet, HiFormer-B, Swin-UNet, SegFormer-B2 from scratch (5 seeds); ImageNet-pretrained DeepLabv3+, HRNet-W32, SegFormer-B2 (3 seeds) |
| **Degradations** | 3 generic (blur, noise, JPEG) + 3 modality-specific per task = 18 physical models, each calibrated per image to 8 SSIM levels |
| **Scale** | 245 networks (8 × 5 tasks × 5 seeds + 3 × 5 tasks × 3 seeds), **6,295,040 stored predictions** (validation, test and external images; best and last checkpoints) |
| **Metrics** | Dice, IoU, precision, recall, HD95, ASSD, normalised surface Dice (NSD), error areas, confidence features |
| **Analysis** | 5 confirmatory tests (Holm), cross-fitted associations, effective robustness, label-free model selection, operating limits |

The five confirmatory hypotheses were fixed and hashed before any test or external image was evaluated:

| | Hypothesis |
|---|---|
| **H1** | Clean Dice predicts degraded Dice (cross-fitted slope across models). |
| **H2** | Effective robustness to generic degradations predicts effective robustness to modality-specific degradations. |
| **H3** | The architecture contributes to the variance of robustness. |
| **H4** | Synthetic effective robustness predicts effective robustness at the external site. |
| **H5** | The primary label-free estimator changes the held-out selection regret relative to clean-Dice selection. |

## Contents

1. [Pipeline](#1-pipeline)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Run](#5-run)
6. [Configuration](#6-configuration)
7. [Outputs](#7-outputs)
8. [Reproducibility](#8-reproducibility)
9. [Released outputs](#9-released-outputs)
10. [License](#10-license)
11. [Citation](#11-citation)

## 1. Pipeline

```
prepare ──> train ──┬──> infer_val ──> freeze ──> infer_test ──> analyze
  │           │     │       │            │            │             │
  │           │     │       │            │            │             └─ confirmatory tests H1-H5, selection, variance,
  │           │     │       │            │            │                operating limits, sensitivity, tables, figures
  │           │     │       │            │            └─ clean, degraded and external test images, every network
  │           │     │       │            └─ label-free estimator chosen on validation; plan, splits, checkpoints,
  │           │     │       │               code hashed before any test image is evaluated
  │           │     │       └─ validation images; validation Dice must reproduce training
  │           │     └─ degrade: strength per image and degradation that gives each target SSIM (runs during training)
  │           └─ 8 architectures (+ 3 pretrained) × 5 tasks × seeds, resumable, best checkpoint by validation Dice
  └─ preprocessed caches of every split, normalisation statistics, pretrained weights
```

Every stage is **safe to rerun**: finished work is skipped, and interrupted training or inference resumes where it
stopped. Before any test prediction is stored, inference must reproduce the validation Dice recorded at the end of
training (`gates.repro_train_tolerance`), and `infer_test` refuses to run until `freeze` has written
`prereg/prereg.json`.

## 2. Repository layout

```
configs/config.yaml               every path and setting of the paper runs
splits/{set}.json                 split manifests of the 5 tasks and 5 external sites
shift_stress_test/
  cli.py, pipeline.py             command line (--stages, --models, --datasets, --seeds, --set, --status), orchestration
  config.py, layout.py, status.py YAML loader with base files and --set overrides, every output path, stage progress
  data/                           native loading, resizing, split caches, normalisation, groups and strata
  models/                         the eight architectures
  training/                       trainer (AdamW, warm-up + cosine, mixed precision, resume), losses, augmentation,
                                  multi-GPU runner
  degrade/                        generic, dermoscopy, endoscopy, ultrasound, fundus and chest X-ray degradations,
                                  field-of-view SSIM, per-image matching to the target SSIM, degraded sets, atlas
  inference/                      batched prediction over every set and level, quality gates
  metrics/                        Dice, IoU, precision, recall, HD95, ASSD, NSD, error areas, confidence features
  analysis/                       robustness scores, cross-fitting, confirmatory tests, mixed models, label-free
                                  selection, operating limits, sensitivity, power, freeze, tables, figures
  utils/                          I/O, logging, job registry, GPU memory, precision, seeding
tests/                            unit tests and a tiny debug configuration (tests/configs/tiny.yaml)
pyproject.toml                    package, dependencies and the shift-stress-test command
requirements.txt                  dependencies with the versions of the paper runs
LICENSE                           MIT license of the code
```

## 3. Installation

```bash
git clone https://github.com/sarang-banakhede/Multi-Modality-Acquisition-and-Site-Shift-Stress-Test.git
cd Multi-Modality-Acquisition-and-Site-Shift-Stress-Test
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest -q                          # unit tests, no data and no GPU needed
```

Python 3.10 or newer. Versions of the paper runs: Python 3.10.12, PyTorch 2.14.1 (CUDA 13.0), MONAI 1.6.0,
timm 1.0.30, segmentation-models-pytorch 0.5.0, einops 0.8.2.

## 4. Data

All datasets are public. The pipeline **expects the data already arranged** in the structure below and does not
rearrange the raw downloads. Download each set from its provider, convert images and masks to PNG without altering
them, and sort them according to the manifests in `splits/`. File names are identical in `images/` and `masks/` and
equal to the `id` of the manifest; masks are binary {0, 255}. Grayscale sets (TN3K, Darwin, DDTI, Montgomery +
Shenzhen) are read as grayscale and repeated to three channels.

```
data/                                  (or any folder: paths.data)
├── isic2016/      {train,val,test}/{images,masks}/
├── kvasir_seg/    {train,val,test}/{images,masks}/
├── tn3k/          {train,val,test}/{images,masks}/   ids prefixed train_ / test_
├── refuge2/       {train,val,test}/{images,masks}/   masks are the neuroretinal rim (disc minus cup)
├── darwin_cxr/    {train,val,test}/{images,masks}/
├── ph2/           {images,masks}/                    external site of isic2016
├── cvc_colondb/   {images,masks}/                    external site of kvasir_seg
├── ddti/          {images,masks}/                    external site of tn3k
├── drishti_gs1/   {images,masks}/                    external site of refuge2, rim from the 4-expert majority vote
└── cxr_mont_shen/ {images,masks}/                    external site of darwin_cxr, ids prefixed mont_ / shen_
```

| Task | Modality, target | Train | Val | Test | Split rule |
|---|---|---|---|---|---|
| ISIC 2016 | dermoscopy, lesion | 787 | 110 | 379 | official test, val 12% of train, 3 duplicates of test removed |
| Kvasir-SEG | endoscopy, polyp | 700 | 100 | 200 | 70 / 10 / 20, stratified |
| TN3K | ultrasound, thyroid nodule | 2532 | 347 | 614 | official test, val 12% of train |
| REFUGE2 | fundus, neuroretinal rim | 700 | 100 | 400 | official test, val stratified by subset and glaucoma label |
| Darwin | chest X-ray, lungs | 4187 | 636 | 1202 | 70 / 10 / 20 by patient, 81 empty masks removed |

| External site | Task | Images |
|---|---|---|
| PH2 | ISIC 2016 | 200 |
| CVC-ColonDB | Kvasir-SEG | 380 |
| DDTI | TN3K | 637 |
| Drishti-GS1 | REFUGE2 | 101 |
| Montgomery + Shenzhen | Darwin | 704 (138 + 566) |

Patients and confirmed near-duplicates (perceptual hash distance at most 8 and band-pass correlation at least 0.6)
never cross a split, and no external image duplicates an image of its task. Each internal manifest lists one item per
image with its `id`, `split`, `group` (patient or near-duplicate cluster) and `stratum` (target-area quartile, plus
subgroup where relevant); an external manifest lists every `id` and any near-duplicate pairs within the set. A new
dataset needs its own manifest in the same format. Everything after that is automatic: `prepare` resizes every image
to 224 × 224, replicates grayscale channels and computes the normalisation statistics on the training split only.

## 5. Run

The whole pipeline, or any part of it:

```bash
shift-stress-test                                            # every stage of configs/config.yaml
shift-stress-test --stages prepare                           # split caches, normalisation, pretrained weights
shift-stress-test --stages train --models unet transunet --datasets kvasir_seg --seeds 0 1
shift-stress-test --stages degrade                           # SSIM-calibrated degraded sets
shift-stress-test --stages infer_val                         # validation inference + reproduction check
shift-stress-test --stages freeze                            # choose the estimator, hash the analysis plan
shift-stress-test --stages infer_test                        # test and external inference
shift-stress-test --stages analyze                           # all statistics, tables and figures
shift-stress-test --status                                   # what is trained, degraded, inferred and analysed
```

Reproduce every statistic of the paper from Part 1 of the [released outputs](#9-released-outputs), on a CPU and
without the data:

```bash
shift-stress-test --stages analyze
```

Options: `--config file.yaml` and `--set key=value` (several allowed), for example
`--set paths.data=/data/segmentation paths.output=/data/run1 train.total_iters=2000`.

* **Hardware.** Training and inference run on CUDA GPUs: `runtime.device: auto` uses every visible GPU (or the list in
  `runtime.gpus`, e.g. `--set runtime.gpus=[0,1]`) with one worker per GPU, sharing a locked job registry so that no
  job runs twice. `prepare`, `degrade` and `analyze` run on CPU; degradation runs in the background during training
  (`runtime.overlap_degrade`). The paper networks were trained on one NVIDIA L4 GPU.
* **Long jobs.** `runtime.time_limit_hours` stops training cleanly before the limit; rerun the same command to resume.
* **Smoke test.** `shift-stress-test --config tests/configs/tiny.yaml` runs every stage on a few images per split and
  writes to `/tmp/shift_stress_test_debug`, never touching the real results.

## 6. Configuration

Every setting of the paper runs lives in `configs/config.yaml`.

| Section | What it controls |
|---|---|
| `paths` | data, split manifests, output, optional cache folder |
| `experiment` | models, tasks and seeds, external site of each task, grayscale and ring-target sets, pretrained arm |
| `data` | image size, classes, optional per-split image limit for debugging |
| `train` | iterations, batch and micro-batch sizes, learning rate and schedule, weight decay, clipping, augmentation per task |
| `metrics` | NSD tolerance |
| `degrade` | SSIM levels, root-finding tolerances, reachability threshold, physical parameters of each degradation |
| `infer` | batch size, sets evaluated with the last checkpoint |
| `gates` | reproduction tolerance |
| `analysis` | random seed, cross-fitting splits, bootstrap and permutation sizes, tolerances of the operating limits |
| `runtime` | device, GPUs, worker processes, background degradation, time limit |
| `stages` | default stage list |

## 7. Outputs

Everything the pipeline writes goes to `paths.output`, by default `Multi-Modality Acquisition and Site Shift Stress
Test Outputs/` in the repository root.

> [!NOTE]
> The released copy of this folder is split into **three Zenodo records** because of its size, so each record holds
> only part of the tree below. Extracting all three into the same location restores this exact structure; see
> [Released outputs](#9-released-outputs).

```
Multi-Modality Acquisition and Site Shift Stress Test Outputs/
├── config_used.yaml                     configuration of the run
├── degrade_config.json                  SSIM levels, reachability threshold, validation reachability
├── cache/                               preprocessed images and masks, normalisation statistics (norm/)
├── train/{model}/{task}/seed{s}/        best.pt, last.pt, config.json, summary.json, train / val logs, DONE
├── degraded/{task}/{val,test}/
│   ├── {code}.npz                       degraded images of every level with achieved SSIM, strength and reachability
│   ├── reachability.csv                 share of images reaching each level, median SSIM and physical strength
│   └── atlas.png                        preview of every degradation and level (val only)
├── eval/{model}/{task}/seed{s}/
│   ├── best/{val,test,external}/
│   │   ├── {condition}.parquet          one row per image and level
│   │   └── {condition}.masks.npz        packed binary predictions (masks, index, level)
│   └── last/{test,external}/            clean and external predictions of the final checkpoint
├── prereg/                              SHA-256 of plan, manifests, checkpoints and code, estimator choice, power check
├── analysis/                            results.json (H1-H5), scores, selection, operating limits, tail risk,
│                                        pretrained arm, sensitivity analyses, per-run summaries, figures/
├── state/                               job registry, event log, quality gates
└── logs/pipeline.log                    log of every stage
```

`{code}` is a degradation (`G1`-`G3` generic; `D` dermoscopy, `E` endoscopy, `U` ultrasound, `F` fundus, `X` chest
X-ray, each 1-3) and `{condition}` is `clean` or a degradation code. Columns of `eval/**/{condition}.parquet`:
`model, family, dataset, seed, ckpt, subset, degradation, image_id, level, target_ssim, achieved_ssim, strength,
physical, dice, iou, precision, recall, specificity, hd95, assd, nsd, area_error, volume_similarity, gt_area,
pred_area, fp_area, fn_area, n_components, n_spurious, detected, empty_pred, conf_fg, msp, msp_fg, entropy, band_conf`.

* `level = 0` is the unmodified image; levels 1-8 are the SSIM targets 0.95 to 0.60.
* `strength` is the calibrated degradation strength and `physical` the same strength in physical units; unreachable
  levels are flagged in the degraded sets and excluded by the reachability rule.
* `hd95` and `assd` are in pixels of the 224 × 224 grid; `nsd` uses a tolerance of 2 pixels.
* `conf_fg`, `msp`, `msp_fg`, `entropy` and `band_conf` are the image-level confidence features of the label-free
  estimators.

## 8. Reproducibility

* Seeds are fixed for training, splits, degradation patterns (seeded by image and degradation), cross-fitting halves,
  bootstrap and permutation tests; results are reproducible up to GPU floating-point differences.
* The analysis plan, the primary label-free estimator, the split manifests, all checkpoints and the code were hashed
  in `prereg/prereg.json` before any test or external image was evaluated. The code was hashed as it ran, under its
  former package name `segstress`; this release is the same code with the package renamed to `shift_stress_test` and
  with an analysis that runs without the raw data (image names from the manifests, reachability from
  `reachability.csv`). No statistical code changed.
* On re-evaluation by the inference pipeline, all 200 from-scratch networks reproduced the validation Dice recorded
  during training to within 3e-4.
* Models are trained **without** any degradation used at test time, deliberately: the study measures the robustness of
  standard training.

## 9. Released outputs

The complete outputs of the paper runs (about 128 GB) are archived on Zenodo in three records under the CC BY-NC 4.0
licence.

> [!IMPORTANT]
> **The outputs folder is split into three parts.** Each record contains a subset of the folder described in
> [section 7](#7-outputs), under the same top folder `Multi-Modality Acquisition and Site Shift Stress Test Outputs/`.
> Extracting all three records into the same location restores the complete structure. **Part 1 alone reproduces
> every statistic of the paper** and runs inference from the best checkpoints.

| Part | Contents | Size | DOI |
|---|---|---|---|
| **1. Results and Best Weights** | `analysis/`, `prereg/`, `state/`, `logs/`, `cache/norm/`, configurations, training summaries and logs, `best.pt` of every network, every `eval/**/*.parquet`, reachability reports and degradation previews | 36.7 GB | [10.5281/zenodo.23967820](https://doi.org/10.5281/zenodo.23967820) |
| **2. Predicted Masks** | every `eval/**/*.masks.npz`; degraded images of Kvasir-SEG and TN3K | 44.1 GB | [10.5281/zenodo.23967824](https://doi.org/10.5281/zenodo.23967824) |
| **3. Last Weights and Degraded Images** | `last.pt` of every network; degraded images of ISIC 2016, REFUGE2 and Darwin | 47.2 GB | [10.5281/zenodo.23967826](https://doi.org/10.5281/zenodo.23967826) |

Place the extracted folder in the repository root, where `paths.output` expects it:

```
Multi-Modality-Acquisition-and-Site-Shift-Stress-Test/        this repository
├── configs/
├── ...
└── Multi-Modality Acquisition and Site Shift Stress Test Outputs/   released outputs
```

The release omits the image caches in `cache/`, which `prepare` rebuilds from the data; only `cache/norm/`
(normalisation statistics) is included. To rerun inference from the released checkpoints, arrange the data as in
[section 4](#4-data) and run `prepare`, `infer_val` and `infer_test`; to retrain from scratch, run `train`.

## 10. License

* **Code:** released under the [MIT License](LICENSE).
* **Released outputs** (checkpoints, predictions, statistics and degraded images): released under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), since several of the datasets permit
  non-commercial use only.
* **Datasets:** the original datasets, and the degraded images derived from them, remain subject to the licences and
  terms of their providers.

## 11. Citation

**Paper.** The citation will be available on publication.

**Released outputs.** If you use the released outputs, please cite the corresponding Zenodo records:

> Banakhede, S. (2026). *Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 1 - Results and Best
> Weights* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23967820
>
> Banakhede, S. (2026). *Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 2 - Predicted Masks*
> (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23967824
>
> Banakhede, S. (2026). *Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 3 - Last Weights and
> Degraded Images* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23967826

```bibtex
@dataset{banakhede2026outputs_part1,
  author    = {Banakhede, Sarang},
  title     = {Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 1 - Results and Best Weights},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23967820},
  url       = {https://doi.org/10.5281/zenodo.23967820}
}

@dataset{banakhede2026outputs_part2,
  author    = {Banakhede, Sarang},
  title     = {Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 2 - Predicted Masks},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23967824},
  url       = {https://doi.org/10.5281/zenodo.23967824}
}

@dataset{banakhede2026outputs_part3,
  author    = {Banakhede, Sarang},
  title     = {Multi-Modality Acquisition and Site Shift Stress Test Outputs Part 3 - Last Weights and Degraded Images},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.23967826},
  url       = {https://doi.org/10.5281/zenodo.23967826}
}
```
