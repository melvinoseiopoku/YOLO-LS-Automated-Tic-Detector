# YOLO-LS Automated Tic Detector

Research workspace for converting Label Studio `videorectangle` annotations into
Ultralytics YOLO datasets and training an automated tic detector.

## Repository Layout

- `YOLO/Training_V1/` - main research workspace with MATLAB conversion and
  analysis scripts, dataset configs, and dataset report figures.
- `YOLO/Training_V1/yd/` - primary YOLO dataset config. Extracted images and
  label files are kept local and ignored by git.
- `YOLO/Training_V1/yolo_dataset/` - earlier/smaller YOLO export config. The
  generated image and label folders are kept local and ignored by git.
- `runs/detect/train2/` - YOLO training plots, metrics, and preview figures.
  Model weights are kept local and ignored by git.
- `docs/setup/` - setup and troubleshooting notes for Label Studio, Docker, CLI,
  and YOLO integration.
- `docs/poster/` - project poster PDF.

## Dataset Snapshot

Primary dataset: `YOLO/Training_V1/yd`

| Split | Images | Label files |
| --- | ---: | ---: |
| train | 7,304 | 7,159 |
| val | 2,823 | 2,808 |
| test | 1,811 | 1,807 |
| total | 11,938 | 11,774 |

Classes:

1. LED
2. Rest
3. Talking
4. Vocal Tic
5. Motor Tic
6. Right Vol Mov
7. Left Vol Mov

Raw videos are stored locally in `YOLO/Training_V1/videos/` but are ignored by
git. Extracted frame datasets and label files are also ignored.

## Reproducing The Dataset

The MATLAB scripts in `YOLO/Training_V1/` expect to be run from that directory
layout. The main inputs are:

- `training.json` - Label Studio export kept locally.
- `videos/` - original MP4 recordings kept locally.

Useful scripts:

- `ls_videorectangle_to_yolo_allframes.m` - exports all frames from Label Studio
  video rectangle annotations into a YOLO folder.
- `ls_videorectangle_to_yolo_dataset.m` - dataset export script snapshot.
- `analyze_tic_dataset.m` - builds dataset summary figures and tables.
- `sanitycheck.m` - visual sanity checks for generated labels.

## Training Artifacts

The latest included training run summary is under `runs/detect/train2/`,
including:

- `results.csv`
- precision/recall/F1/PR plots
- validation batch previews

## Git LFS

This repository tracks figure images and PDFs through Git LFS. Raw videos,
generated frame datasets, label files, annotation exports, and model weights are
kept local unless explicitly approved for upload. After cloning, run:

```bash
git lfs install
git lfs pull
```

to download the large assets.
