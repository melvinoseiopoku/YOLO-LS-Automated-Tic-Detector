# YOLO CLI — Long Video Prediction Upload to Label Studio

## Overview

This document describes the CLI workflow for generating YOLO predictions on long Label Studio video tasks and uploading them back to Label Studio automatically.

**When to use this workflow:**
When Label Studio times out waiting for the Docker ML backend to finish inference on long videos (15+ minutes), use the CLI tool instead. It runs inference locally and uploads predictions to Label Studio after processing completes — no open connection required.

**When to use the Docker backend instead:**
For short videos (under ~10 seconds), the Docker backend with "Retrieve Predictions" in the Data Manager works reliably and requires no extra setup.

---

## Project Context

This workflow is part of the automated tic detection pipeline for Tourette Syndrome (TS) patients receiving DBS therapy. Videos of TS patients are annotated in Label Studio using a YOLO-based model (`best.pt`) trained on motor tic classes.

**Label Studio project:** YOLO Test V2 (Project ID: 4)  
**Model:** `best.pt` — custom-trained YOLOv8 detection model  
**Classes:** LED, Motor Tic, Vocal Tic, Rest, Talking, Left Vol Mov, Right Vol Mov

---

## Repository Structure

```
label-studio-ml-backend/
└── label_studio_ml/
    └── examples/
        └── yolo/
            ├── cli.py          ← the CLI tool
            ├── model.py        ← YOLO model wrapper
            ├── models/         ← model files including best.pt
            ├── requirements.txt
            └── ...
```

The CLI script is at:
```
C:\Users\"username"\label-studio-ml-backend\label_studio_ml\examples\yolo\cli.py
```

---

## Prerequisites

Before running the CLI:

- Label Studio must be running via the **desktop shortcut** (not `label-studio start` from terminal — these are different installations with different databases and tokens)
- Label Studio must be accessible at `http://localhost:8080`
- A valid API token from the active Label Studio instance
- `label-studio-ml` installed in `C:\LabelStudioApp\env` (see First-Time Setup)
- `best.pt` present in the `models/` folder

---

## First-Time Setup

This only needs to be done once. Install `label_studio_ml` into the LabelStudioApp Python environment:

```bash
C:\LabelStudioApp\env\Scripts\pip.exe install -e C:\Users\"username"\label-studio-ml-backend
```

> **Why this environment?** The desktop shortcut runs Label Studio from `C:\LabelStudioApp\env`. The CLI must use the same Python environment so its dependencies are compatible. Running `cli.py` with the miniconda Python will fail with `ModuleNotFoundError: No module named 'rq'` because `rq` uses Unix `fork` which is not available on Windows.

---

## Finding Your API Token

1. Open Label Studio in your browser using the **desktop shortcut**
2. Click your avatar / initials (top right)
3. Go to **Account & Settings**
4. Copy the **Access Token**

> **Important:** Always copy the token from the same Label Studio instance you will point the CLI at. If you start a second Label Studio instance from the terminal, it has a different database and a different token — your real token will return `401 Invalid token` when used against it.

**Validate your token before running the CLI:**

```bash
python -c "import requests; r=requests.get('http://localhost:8080/api/projects', headers={'Authorization':'Token YOUR_TOKEN_HERE'}); print(r.status_code); print(r.text[:300])"
```

- `200` → token is valid
- `401` → wrong token or wrong Label Studio instance

---

## Finding Project and Task IDs

**Project ID** — visible in the browser URL inside your project:
```
http://localhost:8080/projects/4/data  →  project ID is 4
```

**Task ID** — visible in the Data Manager task list (leftmost column), or in the URL when you open a task:
```
http://localhost:8080/tasks/11/  →  task ID is 11
```

---

## Required Environment Variables

These must be set in the **same terminal window** before running the CLI. They are used internally by the Label Studio SDK to download video files from the server — the CLI arguments alone are not sufficient.

```bash
set LABEL_STUDIO_URL=http://localhost:8080
set LABEL_STUDIO_API_KEY=YOUR_TOKEN_HERE
```

---

## Running the CLI

### Full execution sequence (copy-paste ready)

```bash
set LABEL_STUDIO_URL=http://localhost:8080
set LABEL_STUDIO_API_KEY=YOUR_TOKEN_HERE
C:\LabelStudioApp\env\Scripts\python.exe C:\Users\"username"\label-studio-ml-backend\label_studio_ml\examples\yolo\cli.py --ls-url http://localhost:8080 --ls-api-key YOUR_TOKEN_HERE --project 4 --tasks 11
```

### Single task

```bash
... --project 4 --tasks 11
```

### Multiple tasks (sequential, one window)

```bash
... --project 4 --tasks 11,12,13
```

### Multiple tasks (parallel, multiple windows)

Open separate Command Prompt windows and run a different task ID in each:

```
Window 1: ... --tasks 11
Window 2: ... --tasks 12
Window 3: ... --tasks 13
```

> Start with 2 parallel windows and monitor CPU/RAM before adding more.

### Using a JSON file

Pass a `.json` file instead of task IDs:

```bash
... --tasks tasks.json
```

Task IDs only:
```json
[11, 12, 13]
```

Full task data:
```json
[{"id": 11, "data": {"video_url": "/data/upload/4/video.mp4"}}]
```

---

## CLI Parameters

| Parameter | Default | Description |
|---|---|---|
| `--ls-url` | `http://localhost:8080` | URL of the Label Studio instance |
| `--ls-api-key` | None | Label Studio API authentication token (raw token only — do not include the `Token ` prefix) |
| `--project` | `1` | Label Studio project ID |
| `--tasks` | `tasks.json` | Comma-separated task IDs or path to a JSON file |

**Correct token format:**
```
--ls-api-key abc123xyz
```

**Incorrect (do not include the `Token` prefix):**
```
--ls-api-key "Token abc123xyz"
```

---

## How It Works

1. The SDK connects to Label Studio and retrieves the project's labeling config XML
2. Task data (including the `video_url`) is fetched from Label Studio
3. The YOLO model is loaded from `models/best.pt` using settings from the labeling config (`model_path`, `model_tracker`, `model_conf`)
4. The video file is downloaded to a local cache directory using `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY`
5. YOLO runs frame-by-frame inference using the botsort tracker
6. Predictions are formatted into Label Studio's `videorectangle` format with per-frame bounding box sequences
7. Predictions are uploaded to Label Studio via the SDK and appear in the task's Predictions column

---

## Verifying Predictions Uploaded Successfully

After the CLI completes:

1. Go to Label Studio → your project → **Data Manager**
2. Check the **Predictions** column on the task row — it should show a count (e.g., `1`)
3. Click the task to open it — bounding boxes should appear on the video timeline as preannotations

---

## Common Errors and Fixes

### `ModuleNotFoundError: No module named 'label_studio_ml'`

**Cause:** Running `cli.py` with the wrong Python (miniconda instead of LabelStudioApp).  
**Fix:** Use the explicit path to the LabelStudioApp Python:
```bash
C:\LabelStudioApp\env\Scripts\python.exe cli.py ...
```

### `ModuleNotFoundError: No module named 'rq'` (ValueError: cannot find context for 'fork')

**Cause:** You are using the miniconda Python. The `rq` library requires Unix `fork` which does not exist on Windows.  
**Fix:** Same as above — use `C:\LabelStudioApp\env\Scripts\python.exe`.

### `401 Invalid token`

**Cause:** The token is wrong, expired, or belongs to a different Label Studio instance.  
**Fix:**
- Make sure Label Studio is running via the **desktop shortcut**
- Copy a fresh token from Account & Settings in that instance
- Validate the token with the `python -c "import requests..."` command above

### `FileNotFoundError: Can't resolve url ... You can set LABEL_STUDIO_URL environment variable`

**Cause:** `LABEL_STUDIO_URL` is not set as a shell environment variable.  
**Fix:**
```bash
set LABEL_STUDIO_URL=http://localhost:8080
```

### `FileNotFoundError: To access uploaded and local storage files you have to set LABEL_STUDIO_API_KEY`

**Cause:** `LABEL_STUDIO_API_KEY` is not set as a shell environment variable.  
**Fix:**
```bash
set LABEL_STUDIO_API_KEY=YOUR_TOKEN_HERE
```

### `label_studio_sdk.core.api_error.ApiError: status_code: 401`

**Cause:** Multiple Label Studio instances are running (one from desktop shortcut, one from terminal). The CLI is authenticating against the wrong one.  
**Fix:** Close all Label Studio instances. Start only the **desktop shortcut** version. Get a fresh token from that instance.

---

## Important Notes on Label Studio Instances

This machine has **two separate Label Studio installations**:

| | Desktop Shortcut | Terminal (`label-studio start`) |
|---|---|---|
| Location | `C:\LabelStudioApp\env` | `C:\Users\...\miniconda3` |
| Version | 1.13.1 | 1.22.0 |
| Database | `C:\LabelStudioApp\...` | `AppData\Local\label-studio` |
| Projects/data | Your real projects | Empty / separate |

Always use the **desktop shortcut** version. The CLI, Docker backend, and browser must all point to the same instance.

---

## Docker Backend vs CLI — Summary

| | Docker Backend | CLI Tool |
|---|---|---|
| Trigger | Data Manager → Actions → Retrieve Predictions | Command line |
| Connection | Label Studio holds open HTTP connection | No connection held open |
| Timeout risk | Yes — long videos may time out | No |
| Best for | Short videos, quick testing | Long videos (15+ min) |
| Predictions stored | Yes, if response received before timeout | Yes, always |
| Setup required | Docker running, backend connected | Environment variables set |
