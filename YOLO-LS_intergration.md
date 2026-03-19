# Label Studio + Docker + Pretrained YOLO Integration

## Overview

This document describes the complete workflow for connecting **Label Studio** to a **pretrained YOLO model** running inside Docker, so that predictions are returned as preannotations in the Label Studio interface.

This setup is specifically for a **Windows local-development workflow** using:
- Label Studio running on the host machine via desktop shortcut
- Docker Desktop running the YOLO ML backend
- A custom pretrained YOLO checkpoint (`best.pt`)
- Video tasks in Label Studio

For long videos that exceed Label Studio's request timeout, see the companion document: **CLI_README.md**.

---

## Architecture

```
Browser (you)
     │
     ▼
Label Studio  ←──── localhost:8080
     │
     │  POST /predict (task data + labeling config)
     ▼
Docker YOLO Backend ←──── localhost:9090
     │
     │  fetches video file from Label Studio
     │  using host.docker.internal:8080
     ▼
YOLO inference (best.pt)
     │
     │  returns predictions (per-frame bounding boxes)
     ▼
Label Studio displays preannotations
```

**Key networking rule:**
- Label Studio → Docker: `http://localhost:9090`
- Docker → Label Studio: `http://host.docker.internal:8080` (never `localhost` inside a container)

---

## Directory Layout

```
AutonomousTicDetection/
├── docker/
│   └── docker-compose.yml
├── models/
│   └── best.pt              ← your custom model goes here
├── data/
│   └── server/              ← used internally by the backend
├── cache_dir/               ← video cache used during inference
├── configs/
│   └── label_studio_video_config.xml
└── notes/
    └── setup_log.md
```

---

## Prerequisites

- Label Studio running via **desktop shortcut** at `http://localhost:8080`
- Docker Desktop installed and running
- A valid Label Studio API token
- `best.pt` placed in the `models/` folder
- A Label Studio project with video tasks already imported

> **Critical:** Always start Label Studio using the **desktop shortcut**, not `label-studio start` from a terminal. These are two separate installations with separate databases and separate tokens. Docker, the browser, and any CLI tools must all point to the same running instance.

---

## Step 1 — Start Label Studio

Use your desktop shortcut to start Label Studio. Confirm it loads at `http://localhost:8080` and your project is visible.

Then get your API token:
1. Click your avatar/initials (top right)
2. Go to **Account & Settings**
3. Copy the **Access Token**

**Validate the token before proceeding:**

```bat
python -c "import requests; r=requests.get('http://localhost:8080/api/projects', headers={'Authorization':'Token YOUR_TOKEN_HERE'}); print(r.status_code); print(r.text[:300])"
```

- `200` → token is valid, continue
- `401` → wrong token or wrong instance, do not continue

---

## Step 2 — Place Your Model

Copy `best.pt` into the `models/` folder next to `docker-compose.yml`:

```
AutonomousTicDetection/models/best.pt
```

The Docker volume mount `./models:/app/models` maps this to `/app/models/best.pt` inside the container. The labeling config references it as `model_path="best.pt"`, which the backend resolves relative to `MODEL_ROOT=/app/models`.

---

## Step 3 — Configure docker-compose.yml

```yaml
version: "3.8"

services:
  yolo:
    container_name: yolo
    image: humansignal/yolo:v0
    build:
      context: .
      args:
        TEST_ENV: ${TEST_ENV}
    environment:
      - BASIC_AUTH_USER=
      - BASIC_AUTH_PASS=
      - LOG_LEVEL=DEBUG
      - WORKERS=1
      - THREADS=8
      - MODEL_DIR=/data/models
      - PYTHONPATH=/app

      # Label Studio connection — MUST use host.docker.internal, not localhost
      - LABEL_STUDIO_URL=http://host.docker.internal:8080
      - LABEL_STUDIO_API_KEY=YOUR_TOKEN_HERE

      # YOLO settings
      - ALLOW_CUSTOM_MODEL_PATH=true
      - DEBUG_PLOT=false
      - MODEL_SCORE_THRESHOLD=0.5
      - MODEL_ROOT=/app/models

    extra_hosts:
      - "host.docker.internal:host-gateway"

    ports:
      - "9090:9090"

    volumes:
      - "./data/server:/data"
      - "./models:/app/models"
      - "./cache_dir:/app/cache_dir"
```

### Key settings explained

| Setting | Why it matters |
|---|---|
| `LABEL_STUDIO_URL=http://host.docker.internal:8080` | Inside Docker, `localhost` refers to the container itself. `host.docker.internal` reaches the Windows host where Label Studio is running. |
| `LABEL_STUDIO_API_KEY` | Used by the backend to authenticate and download video files from Label Studio storage. |
| `ALLOW_CUSTOM_MODEL_PATH=true` | Allows the backend to load a custom model file specified in the labeling config XML. |
| `MODEL_ROOT=/app/models` | The root directory inside the container where model files are looked up. |
| `./models:/app/models` | Mounts your local `models/` folder into the container. |
| `LOG_LEVEL=DEBUG` | Shows full request/response logs — essential during setup. |
| `WORKERS=1` | Recommended during setup to keep logs readable. |

---

## Step 4 — Start the Docker Backend

From the folder containing `docker-compose.yml`:

```bat
docker compose down
docker compose up
```

**Healthy startup looks like:**

```
[INFO] Listening at: http://0.0.0.0:9090
[INFO] Loading yolo model: /app/models/best.pt
[INFO] Model /app/models/best.pt names: {0: 'LED', 1: 'Rest', ...}
{"model_class":"YOLO","status":"UP"}
```

> If you only see `status: UP` in the logs with no model loading lines, Docker is running but may not have loaded `best.pt` yet. The model loads on the first prediction request.

> If Docker crashes immediately, check for Windows line-ending issues in config files (CRLF vs LF).

---

## Step 5 — Configure the Label Studio Labeling Interface

In Label Studio → your project → **Settings** → **Labeling Interface** → **Code**, paste:

```xml
<View>
  <Header value="Video Tracking Tool for Tic Marking"/>

  <Labels name="videoLabels" toName="video" allowEmpty="true">
    <Label value="LED" background="#E3170A"/>
    <Label value="Motor Tic" background="#8A4FFF"/>
    <Label value="Vocal Tic" background="#8B95C9"/>
    <Label value="Rest" background="#2292A4"/>
    <Label value="Talking" background="#E43F6F"/>
    <Label value="Left Vol Mov" background="#FF7733"/>
    <Label value="Right Vol Mov" background="#60A561"/>
  </Labels>

  <Video name="video" value="$video_url" sync="audio" framerate="30.0" />

  <VideoRectangle
    name="box"
    toName="video"
    model_path="best.pt"
    model_tracker="botsort"
    model_conf="0.25"
  />

  <Audio name="audio" value="$video_url" sync="video" speed="false"/>
</View>
```

### Critical XML rules

| Element | Rule |
|---|---|
| `<Labels toName="video">` | Must point to `"video"` (the Video tag), NOT to `"box"`. Pointing to `"box"` causes: `Exception: Multiple object tags connected`. |
| `value="$video_url"` | Task JSON must contain a field named exactly `video_url`. Any other field name will silently fail. |
| `model_path="best.pt"` | Must match the filename inside `MODEL_ROOT`. Requires `ALLOW_CUSTOM_MODEL_PATH=true`. |
| `<Audio>` | Do not add `toName` to the Audio tag. It is an object tag, not a control tag. |

---

## Step 6 — Connect the Backend to Label Studio

In Label Studio → your project → **Settings** → **Model**:

1. Click **Add Model** (or edit existing)
2. Set URL to: `http://localhost:9090`
3. Save

Then go to **Settings** → **Annotation**:

1. Enable **Use predictions to prelabel tasks**
2. Select **Yolo (Connected)** from the dropdown
3. Save

> Leave **Interactive Preannotations** OFF. Preannotations for video tasks are triggered manually via the Data Manager.

---

## Step 7 — Trigger Predictions

1. Go to your project's **Data Manager** (task list view)
2. Check the checkbox next to one or more tasks
3. Click **Actions** → **Retrieve Predictions**
4. Watch the Docker logs in real time

**What healthy Docker logs look like during prediction:**

```
[INFO] Run prediction on 1 tasks, project ID = 4
[INFO] Loading yolo model: /app/models/best.pt
[INFO] Model /app/models/best.pt names: {0: 'LED', 1: 'Rest', ...}
video 1/1 (frame 1/177) ...: 384x640 1 Motor Tic, 185.6ms
video 1/1 (frame 2/177) ...: 384x640 1 Motor Tic, 201.7ms
...
Response status: 200 OK
```

After predictions complete, open the task — bounding boxes will appear on the video timeline.

---

## Step 8 — Verify the Correct Model Is Being Used

The Docker logs will show the model class names when `best.pt` loads:

```
Model /app/models/best.pt names:
{0: 'LED', 1: 'Rest', 2: 'Talking', 3: 'Vocal Tic', 4: 'Motor Tic', 5: 'Right Vol Mov', 6: 'Left Vol Mov'}
```

If you see COCO class names (`person`, `bicycle`, `car`, ...) instead, the default YOLOv8 model is being used, not `best.pt`. Check that:
- `best.pt` exists in `./models/`
- `ALLOW_CUSTOM_MODEL_PATH=true` is set
- `model_path="best.pt"` is in the labeling config XML

---

## Long Video Limitation

For videos longer than ~10 seconds, Label Studio may time out waiting for Docker to complete inference. The symptoms are:
- Docker logs show active processing
- Label Studio shows no preannotations after the video loads
- No predictions appear in the Data Manager

This is a **timeout limitation**, not a setup failure. The Docker backend is working correctly.

**Solution:** Use the CLI workflow for long videos. See **CLI_README.md**.

---

## Common Errors and Fixes

### `Exception: Multiple object tags connected, you should specify name`

**Cause:** `<Labels toName="box">` — Labels is pointing to the VideoRectangle control tag instead of the Video object tag.  
**Fix:** Change to `<Labels toName="video">`.

---

### Backend shows `status: UP` but no preannotations appear

**Cause:** Label Studio is only polling the `/health` endpoint. The `/predict` endpoint is never being called.  
**Fixes:**
- Confirm the backend is connected in **Settings → Model**
- Confirm **Use predictions to prelabel tasks** is enabled in **Settings → Annotation**
- Trigger predictions via **Data Manager → Actions → Retrieve Predictions** (do not just open a task)
- Test on a short video first

---

### `401 Invalid token` in Docker logs

**Cause:** Wrong token, stale token, or token from a different Label Studio instance.  
**Fix:** Copy a fresh token from Account & Settings. Update `LABEL_STUDIO_API_KEY` in `docker-compose.yml`. Run `docker compose down` then `docker compose up`.

---

### Uploaded file paths cannot be resolved

**Cause:** `LABEL_STUDIO_URL` is set to `localhost` inside Docker.  
**Fix:** Use `http://host.docker.internal:8080`, not `http://localhost:8080`, in `docker-compose.yml`.

---

### `best.pt` not found / wrong model loads

**Cause:** File missing from `./models/`, or mount path mismatch.  
**Checklist:**
- `./models/best.pt` exists on host
- `docker-compose.yml` has `- "./models:/app/models"`
- `MODEL_ROOT=/app/models` is set
- `ALLOW_CUSTOM_MODEL_PATH=true` is set
- Labeling config has `model_path="best.pt"`

---

### `exec /app/start.sh: No such file or directory`

**Cause:** Windows CRLF line endings in shell scripts.  
**Fix:** Before cloning the repo, run:
```bat
git config --global core.autocrlf false
```
Then re-clone.

---

## Validation Checklist

Before calling the setup complete, verify each item:

- [ ] Label Studio loads at `http://localhost:8080` via desktop shortcut
- [ ] API token returns `200` from the validation command
- [ ] `best.pt` is in `./models/`
- [ ] `docker-compose.yml` uses `host.docker.internal:8080` for `LABEL_STUDIO_URL`
- [ ] `docker compose up` starts without errors
- [ ] Docker logs show `status: UP`
- [ ] Label Studio shows backend as **Connected** in Settings → Model
- [ ] Labeling config uses `<Labels toName="video">` (not `toName="box"`)
- [ ] Task JSON contains `video_url` field matching `$video_url` in config
- [ ] Short video test: Docker logs show inference activity after Retrieve Predictions
- [ ] Short video test: preannotations appear in Label Studio
- [ ] Docker logs show `best.pt` class names (not COCO classes)

---

## Working Configuration Summary

| Component | Value |
|---|---|
| Label Studio URL (host) | `http://localhost:8080` |
| Docker backend URL (from host) | `http://localhost:9090` |
| Docker → Label Studio URL | `http://host.docker.internal:8080` |
| Model host path | `./models/best.pt` |
| Model container path | `/app/models/best.pt` |
| Labeling config field | `$video_url` |
| Task JSON field | `video_url` |
| Tracker | `botsort` |
| Confidence threshold | `0.25` |

---

## When to Use Docker vs CLI

| | Docker Backend | CLI Tool |
|---|---|---|
| Best for | Short videos, testing, validation | Long videos (15+ min), batch runs |
| Timeout risk | Yes | No |
| Trigger method | Data Manager → Retrieve Predictions | Command line |
| Setup needed | Docker running, backend connected | Environment variables set |
| See | This document | CLI_README.md |
