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

The image is built for Linux CPU. A push to `main` hits a webhook on Fox, which runs `git pull` and `docker compose up -d --build`. Tailscale Funnel publishes that webhook at `https://fox.tailcc00e1.ts.net/mediko-deploy`. The dashboard itself stays on port `8501` and is not on the public Funnel URL.

The root of `https://fox.tailcc00e1.ts.net` already belongs to another service. The MediKo hook is only the `/mediko-deploy` path.

Rebuild log on Fox:

```bash
tail -f ~/.local/share/mediko/deploy.log
```

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
