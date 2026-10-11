# MediKo

Research prototype for extracting text from medical reports and, later, measuring biomedical NER on noisy text. This repository is the boilerplate: input handling, embedded PDF text, a PaddleOCR fallback, a CPU Docker image, and a deploy path onto Fox.

It does not diagnose conditions, recommend treatment, or support medical decisions. Normalization, NER, negation, relations, and the timeline are not implemented. Those stages are marked `unavailable` in the JSON result. The dashboard does not fill them in.

The full research plan is in `medical_report_intelligence_spec.md`.

## What runs today

1. Accept a PDF or UTF-8 TXT file up to 20 MB.
2. Read embedded text from each PDF page.
3. When a page has fewer than 40 stripped characters, rasterize it at 200 DPI and run PaddleOCR 3.x.
4. Return a versioned JSON document (`schema_version` `0.1.0`) and show it in Streamlit.

OCR line scores are PaddleOCR recognition scores. They are labeled `model_confidence` and are not calibrated clinical probabilities.

A digital page that also contains a scanned figure stays on the embedded-text path. This build does not split mixed layouts.

The first scanned page downloads the English OCR models into a Docker volume on Fox. Later pages reuse that volume.

## Fox

The image is built for Linux CPU. GitHub Actions does not build it. A push to `main` copies the source to Fox over SSH and runs `docker compose up -d --build` there.

One-time setup on Fox:

1. Install Docker Engine and the Docker Compose plugin.
2. Add the deploy user to the `docker` group.
3. Put an SSH public key for that user in `authorized_keys`.

GitHub repository secrets for `aibelbin/MediKo`:

| Secret | Required | Purpose |
| --- | --- | --- |
| `FOX_HOST` | yes | Hostname or IP of Fox |
| `FOX_USER` | yes | SSH user that can run Docker |
| `FOX_SSH_KEY` | yes | Private key matching that user's `authorized_keys` entry |
| `FOX_PORT` | no | SSH port, default `22` |
| `FOX_DEPLOY_PATH` | no | Directory on Fox, relative to the user's home, default `mediko`. An absolute path also works. |
| `FOX_KNOWN_HOSTS` | no | Pinned `known_hosts` line for Fox. When unset, the runner accepts the host key on first connect. |

After those secrets exist, a push to `main` deploys. You can also run the **Deploy to Fox** workflow by hand.

The app listens on port `8501` on Fox. The first image build is large because it installs PaddlePaddle and PaddleOCR. The first scanned document then downloads model weights.

Logs on Fox, from the deploy directory:

```bash
docker compose logs -f mediko
```

`config/default.yaml` holds the upload limit, the embedded-text threshold, the OCR language, the device, and the render DPI. `MEDIKO_OCR_DEVICE`, `MEDIKO_OCR_LANG`, and `MEDIKO_MAX_UPLOAD_BYTES` override those values. This image is CPU-only (`cpu`). A GPU build would need a different Paddle wheel and `gpu:0`.

## Layout

```text
app/streamlit_app.py          dashboard
src/mediko/pipeline.py        sequential entry point
src/mediko/extraction/        validation, embedded text, PaddleOCR
config/default.yaml           thresholds and OCR settings
Dockerfile                    CPU image
docker-compose.yml            Fox service
.github/workflows/deploy-fox.yml
```

CLI inside the container:

```bash
python -m mediko /path/to/report.pdf -o /tmp/result.json
```
