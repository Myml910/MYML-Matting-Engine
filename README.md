# MYML-Matting-Engine

Local image background removal and foreground matting engine for future MYML Canvas integration.

Phase 1 provides a runnable FastAPI service, CLI tools, model routing, unified image IO, postprocessing, RGBA output, mask output, and compare grids. The `dummy` backend uses a deterministic ellipse foreground mask; `inspyrenet` uses the real InSPyReNet model through `transparent-background`.

v1.5 adds one experimental model name, `inspyrenet_ben2`. This is not a three-model system: Inspyrenet remains the main mask, and BEN2 is used only to replace or blend the alpha inside an edge band around the Inspyrenet foreground.

## Requirements

- Python 3.10+
- Windows, macOS, or Linux

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes `transparent-background`, which wraps InSPyReNet for the real `inspyrenet` backend.

## Start API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/api/health
```

Remove background and save a transparent PNG:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/matting/remove-bg" `
  -F "file=@test_assets/input/sample.jpg" `
  -F "model=dummy" `
  -F "mode=rgba" `
  -F "output_mask=false" `
  --output outputs/sample_rgba.png
```

Return only the mask:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/matting/remove-bg" `
  -F "file=@test_assets/input/sample.jpg" `
  -F "model=dummy" `
  -F "mode=mask" `
  -F "output_mask=true" `
  --output outputs/sample_mask.png
```

Available model names:

- `dummy`
- `auto`
- `inspyrenet`
- `inspyrenet_ben2`
- `birefnet`
- `ben2`

## CLI

Run a single backend:

```powershell
python scripts/run_single.py `
  --input test_assets/input/sample.jpg `
  --model dummy `
  --mode full_foreground `
  --output outputs/sample_rgba.png `
  --mask-output outputs/sample_mask.png
```

This writes:

- `outputs/sample_rgba.png`
- `outputs/sample_mask.png`

`--input` and `--output` are required. `--mask-output` is optional. Supported CLI model names are `dummy`, `auto`, `inspyrenet`, `inspyrenet_ben2`, `birefnet`, and `ben2`; supported CLI modes are `full_foreground`, `main_subject`, `soft_matting`, and `hard_mask`.

Run the real Inspyrenet backend:

```powershell
python scripts/run_single.py `
  --input test_assets/input/demo.png `
  --model inspyrenet `
  --mode full_foreground `
  --output outputs/demo_inspyrenet_rgba.png `
  --mask-output outputs/demo_inspyrenet_mask.png
```

The first `inspyrenet` run may download model files through `transparent-background`. CPU inference can be slow.

Run the v1.5 experimental Inspyrenet + BEN2 edge enhancement:

```powershell
$env:BEN2_MODEL_PATH = "models_cache/BEN2_Base.pth"
python scripts/run_single.py `
  --input test_assets/input/demo.png `
  --model inspyrenet_ben2 `
  --mode full_foreground `
  --output outputs/demo_inspy_ben2_rgba.png `
  --mask-output outputs/demo_inspy_ben2_mask.png `
  --edge-radius 8 `
  --blend 1.0
```

BEN2 is lazy-loaded only when `model=ben2` or `model=inspyrenet_ben2` is used. The default lookup path is `models_cache/BEN2_Base.pth`; override it with `BEN2_MODEL_PATH`. The BEN2 backend expects the official BEN2 Python code to be installed or available on `PYTHONPATH` so it can call `BEN_Base().loadcheckpoints(...)`. If the BEN2 weights are missing, `dummy`, `auto`, and `inspyrenet` still work.

Compare the default v1.5 set (`dummy`, `inspyrenet`, `inspyrenet_ben2`):

```powershell
python scripts/compare_models.py test_assets/input/demo.png --output-dir outputs/compare_v15 --edge-radius 8 --blend 1.0
```

This writes per-model RGBA and mask PNG files plus:

- `outputs/compare_v15/demo_compare_grid.png`
- `outputs/compare_v15/report.json`

If BEN2 weights are missing, `inspyrenet_ben2` is recorded as an error in `report.json`, and the compare run continues.

## API

### `GET /api/health`

Returns service status, configured device, and available model names.

Example response:

```json
{
  "status": "ok",
  "service": "MYML-Matting-Engine",
  "device": "cpu",
  "available_models": ["dummy", "auto", "inspyrenet", "inspyrenet_ben2", "birefnet", "ben2"]
}
```

### `POST /api/matting/remove-bg`

Consumes `multipart/form-data`.

Fields:

- `file`: image file
- `model`: one of `dummy`, `auto`, `inspyrenet`, `inspyrenet_ben2`, `birefnet`, `ben2`
- `mode`: `rgba` or `mask`
- `output_mask`: boolean; when true, returns a grayscale mask PNG
- `edge_radius`: optional integer for `inspyrenet_ben2`; default `8`
- `blend`: optional float for `inspyrenet_ben2`; default `1.0`

Returns `image/png` as a streaming response.

Use Inspyrenet through the same API request format:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/matting/remove-bg" `
  -F "file=@test_assets/input/demo.png" `
  -F "model=inspyrenet" `
  -F "mode=rgba" `
  -F "output_mask=false" `
  --output outputs/demo_inspyrenet_api_rgba.png
```

Use the v1.5 BEN2 edge enhancement through the same endpoint:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/matting/remove-bg" `
  -F "file=@test_assets/input/demo.png" `
  -F "model=inspyrenet_ben2" `
  -F "mode=rgba" `
  -F "output_mask=false" `
  -F "edge_radius=8" `
  -F "blend=1.0" `
  --output outputs/demo_inspy_ben2_api_rgba.png
```

## Notes

- `inspyrenet` lazy-loads on first use, so service startup does not load model weights.
- The first `inspyrenet` request may download model files; CPU inference can be slow.
- `auto` currently maps to `inspyrenet`; it does not enable BEN2 enhancement by default.
- `ben2` requires `BEN2_MODEL_PATH` to point to `BEN2_Base.pth`.
- BEN2 also requires the official BEN2 Python package or repo code to be available in the Python environment.
- `birefnet` is not supported in v1.5.
- Input images are normalized through Pillow and output as standard RGBA PNG or grayscale mask PNG.
