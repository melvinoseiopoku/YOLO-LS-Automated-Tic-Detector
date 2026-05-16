# Troubleshooting Guide: Docker + Label Studio Integration for a Pretrained YOLO Model

## Overview

This document provides a focused troubleshooting guide for integrating **Label Studio (LS)** with a **pretrained YOLO model through Docker**. It is intended for situations where:
- the Docker backend starts but does not return predictions,
- Label Studio connects to the backend but shows no preannotations,
- custom model loading through `best.pt` fails,
- backend health checks work but actual inference does not,
- long video tasks do not show predictions in the interface.

This guide covers **Docker + Label Studio only**. It does **not** cover the separate CLI workflow for asynchronous long-video prediction upload.

---

## Core Principle

A working Docker + Label Studio integration requires **all four** of the following to be correct at the same time:

1. **The correct Label Studio instance is running**
2. **Docker is configured to reach that exact Label Studio instance**
3. **The pretrained model file is mounted and accessible inside Docker**
4. **The Label Studio XML config matches both the task format and backend expectations**

If any one of these is wrong, the backend may still appear healthy while predictions fail.

---

## Quick Diagnostic Checklist

Before diving into specific errors, verify the following:

- [ ] Label Studio is running on the expected host and port
- [ ] You are using the same Label Studio instance in browser and Docker
- [ ] Your Label Studio token is valid
- [ ] Docker uses `host.docker.internal`, not `localhost`, to reach Label Studio
- [ ] Your custom model file `best.pt` exists in the host `models/` directory
- [ ] That directory is mounted into Docker
- [ ] `ALLOW_CUSTOM_MODEL_PATH=true` is set
- [ ] The project XML uses the correct task field, such as `$video_url`
- [ ] The task JSON actually contains that field
- [ ] The model is connected in Label Studio under **Settings → Machine Learning**
- [ ] You have tested with a short video before testing a long video

---

## Symptom 1: Docker Starts Successfully, But No Preannotations Appear

### Typical signs
- Docker container starts
- Label Studio connects to `http://localhost:9090`
- Opening a task does not show predictions
- Docker logs may only show health checks

### Most likely causes
1. Label Studio is only checking backend health and not completing prediction requests
2. The XML labeling config is incompatible with the task type
3. The task data field does not match the XML field name
4. The task is a long video and the request times out before results are returned

### What to check
- Confirm that your XML uses the correct task key:
  - If the XML says `value="$video_url"`, then the task JSON must contain `video_url`
- Confirm that the backend logs show actual inference activity, not only status checks
- Test with a shorter video first
- Confirm the backend is connected to the right project in Label Studio

### Important distinction
If Docker logs only show:

```text
{"model_class":"YOLO","status":"UP"}
```

that means the backend is alive. It does **not** mean it is successfully predicting.

---

## Symptom 2: Docker Backend Is Reachable, But Label Studio Shows `status: UP` Only

### Meaning
This means Label Studio can contact the backend’s health endpoint, but inference may not be running or predictions may not be returning.

### Likely causes
- Task configuration mismatch
- Label Studio not triggering prediction properly
- Video prediction request timing out
- Model file not loading correctly

### Resolution steps
1. Open a short task first, not a long one
2. Watch the Docker logs while opening the task
3. Look for evidence of:
   - media download,
   - model loading,
   - actual inference,
   - prediction response generation
4. Verify the XML config is valid for video tasks

---

## Symptom 3: `Invalid token` or Authentication Errors in Docker

### Meaning
Docker is trying to call Label Studio but is not authenticated successfully.

### Common causes
- Wrong token copied from Label Studio
- Token regenerated but Docker still uses the old one
- Docker is pointing to a different Label Studio instance than the browser
- Wrong Label Studio host/port

### Resolution steps
1. In the browser, open the exact Label Studio instance you are using
2. Go to **Account & Settings**
3. Copy a fresh Access Token
4. Update the Docker compose file:

```yaml
- LABEL_STUDIO_API_KEY=YOUR_NEW_TOKEN_HERE
```

5. Rebuild and restart Docker:

```bat
docker compose down
docker compose up --build
```

### Validation test
Before debugging Docker, validate the token directly:

```bat
python -c "import requests; headers={'Authorization':'Token YOUR_TOKEN_HERE'}; r=requests.get('http://localhost:8080/api/projects', headers=headers); print(r.status_code); print(r.text[:300])"
```

- `200` means the token is valid
- `401` means the token is wrong for that Label Studio instance

---

## Symptom 4: Docker Cannot Reach Label Studio

### Common mistake
Using:

```yaml
- LABEL_STUDIO_URL=http://localhost:8080
```

inside Docker.

### Why this fails
Inside the container, `localhost` refers to the container itself, not the host machine.

### Correct setting on Windows
Use:

```yaml
- LABEL_STUDIO_URL=http://host.docker.internal:8080
```

### Additional note
From the browser and Label Studio side, the Docker backend is still reached through:

```text
http://localhost:9090
```

So:
- **Host → Docker:** `localhost:9090`
- **Docker → Host Label Studio:** `host.docker.internal:8080`

---

## Symptom 5: `best.pt` Does Not Load or Model Seems Ignored

### Meaning
The backend is either not seeing your custom checkpoint or is falling back to another model behavior.

### Common causes
- `best.pt` is not in the mounted `models/` folder
- The Docker volume mount is wrong
- `ALLOW_CUSTOM_MODEL_PATH=true` is missing
- `MODEL_ROOT` does not match the mounted directory
- The XML uses a model name that cannot be resolved

### Resolution steps
1. Confirm the file exists on the host:
   ```text
   ./models/best.pt
   ```

2. Confirm the Docker compose file contains:
   ```yaml
   - "./models:/app/models"
   ```

3. Confirm the environment contains:
   ```yaml
   - ALLOW_CUSTOM_MODEL_PATH=true
   - MODEL_ROOT=/app/models
   ```

4. Confirm the Label Studio XML contains:
   ```xml
   model_path="best.pt"
   ```

### Practical verification strategy
Temporarily rename or remove `best.pt`.
- If behavior changes or Docker errors, the model path is active.
- If nothing changes, the backend may not be using the custom model path correctly.

---

## Symptom 6: Label Studio XML Errors or Invalid Interface Tags

### Meaning
Label Studio rejects the project configuration or silently fails to behave as expected.

### Common causes
- Using image-only tags for video tasks
- Nesting labels under unsupported controls
- Using the wrong target in `toName`

### Working pattern for this project
A valid pattern for video tracking in this project is:

```xml
<View>
  <Header value="Video Tracking Tool for Tic Marking"/>

  <Video name="video" value="$video_url" sync="audio" framerate="30.0" />

  <VideoRectangle
    name="box"
    toName="video"
    model_path="best.pt"
    model_tracker="botsort"
    model_conf="0.25"
  />

  <Labels name="videoLabels" toName="box" allowEmpty="true">
    <Label value="LED" background="#E3170A"/>
    <Label value="Motor Tic" background="#8A4FFF"/>
    <Label value="Vocal Tic" background="#8B95C9"/>
    <Label value="Rest" background="#2292A4"/>
    <Label value="Talking" background="#E43F6F"/>
    <Label value="Left Vol Mov" background="#FF7733"/>
    <Label value="Right Vol Mov" background="#60A561"/>
  </Labels>

  <Audio name="audio" value="$video_url" sync="video" speed="false"/>
</View>
```

### Important note
The XML, task JSON, and model labels must all agree.

If your XML uses `$video_url`, then the task data must provide `video_url`.

---

## Symptom 7: Task Loads, But Docker Cannot Resolve the Video File

### Meaning
The backend received the task but cannot resolve the uploaded media location.

### Common causes
- Relative storage URL cannot be resolved
- Wrong Label Studio URL in Docker
- Invalid token prevents file download
- Label Studio storage is not reachable from the container

### Resolution steps
1. Confirm Docker uses:
   ```yaml
   - LABEL_STUDIO_URL=http://host.docker.internal:8080
   ```

2. Confirm `LABEL_STUDIO_API_KEY` is valid
3. Confirm the task JSON contains a valid `video_url`
4. Open the task in Label Studio and make sure the video loads in the browser first

If the browser cannot load the task video, Docker will not be able to process it either.

---

## Symptom 8: Everything Works for Short Videos But Not Long Videos

### Meaning
This is usually not a setup failure. It is often a timeout limitation.

### Typical pattern
- Short videos: predictions appear
- Long videos: Docker appears to process, but no preannotations appear in Label Studio

### Cause
The backend may complete inference, but Label Studio times out before receiving the full response.

### Recommendation
This is expected behavior for long videos in many cases. Use Docker for:
- connectivity checks,
- custom model loading checks,
- short-video interactive prediction.

Use the separate CLI workflow for long videos.

---

## Symptom 9: Multiple Label Studio Installations or Version Confusion

### Meaning
You may be running more than one Label Studio installation, such as:
- a terminal-based install from Conda
- a desktop shortcut using another Python environment

### Why this matters
Even if they point to the same database, different versions and startup methods can create confusion around:
- ports,
- tokens,
- backend compatibility,
- expected behavior.

### Resolution strategy
Pick **one Label Studio launch method** and use it consistently for:
- browser access,
- Docker integration,
- troubleshooting,
- token generation.

Do not switch back and forth while debugging.

---

## Symptom 10: Docker Config Looks Correct, But Predictions Still Fail

### Escalation checklist
If all obvious settings look correct, verify the following one by one:

1. **Label Studio token works directly**
2. **Label Studio project loads in browser**
3. **Task video loads in browser**
4. **Docker container starts cleanly**
5. **Docker health endpoint returns `status: UP`**
6. **Short task triggers backend activity**
7. **Model file is present and mounted**
8. **XML field name matches task data key**
9. **Project is connected to the backend in Label Studio**
10. **Long-video failures are not misdiagnosed as configuration failures**

---

## Recommended Debugging Order

When troubleshooting, always go from simplest to most complex.

### Step 1
Verify Label Studio is running and the project loads in the browser.

### Step 2
Verify the API token works against the running Label Studio instance.

### Step 3
Verify Docker starts without crashing.

### Step 4
Verify Label Studio connects to the Docker backend.

### Step 5
Verify the backend is receiving real task requests, not only health checks.

### Step 6
Verify the task media loads.

### Step 7
Verify the custom model file is mounted and accessible.

### Step 8
Test on a short video.

### Step 9
Only after all of the above should you interpret long-video failures as timeout issues.

---

## Best Practices

- Use one Label Studio instance consistently
- Keep Docker logs visible during testing
- Start with short videos
- Use `LOG_LEVEL=DEBUG`
- Keep all configuration files under version control
- Document each working change in a setup log
- Separate “integration works” from “long-video UI timeouts”

---

## Summary

Most Docker + Label Studio failures do **not** come from Docker itself. They usually come from one of four sources:
1. wrong Label Studio instance or token,
2. incorrect Docker-to-host URL,
3. missing or inaccessible custom model file,
4. mismatch between the XML config and task data.

Once those are aligned, short-video preannotations should work reliably. If long videos still fail to display predictions, that is most likely a Label Studio timeout limitation rather than a broken Docker integration.
