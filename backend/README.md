# FTIR Backend (AlexNet / LeNet5)

This folder now contains the production FastAPI service that powers Step 4 of the redesigned FTIR workflow. The backend loads the exported `.h5` classifiers from `backend/model/`, runs them with **Keras 3.10 + JAX backend**, and always reports Pearson correlation scores against the clean reference spectra (`SynCleanSet.npy`).

## 1. Environment setup

```bash
cd "/Users/admin/Downloads/FTIR project/backend"

# 1) Create / activate a dedicated venv
python3 -m venv venv
source venv/bin/activate

# 2) Install dependencies (Keras 3 + JAX + FastAPI stack)
pip install --upgrade pip
pip install -r requirements.txt
```

> **Why JAX?**  
> The provided `.h5` weights were saved with Keras 3.10.  
> Running them requires a modern backend (JAX, Torch, or TF 2.16+).  
> JAX offers the smallest CPU-only footprint on macOS, so we use it here.

## 2. Running the API

```bash
cd "/Users/admin/Downloads/FTIR project/backend"
source venv/bin/activate
KERAS_BACKEND=jax uvicorn main:app --reload --port 8000
```

Key environment variables:
- `KERAS_BACKEND=jax` – ensures Keras boots with the JAX runtime.
- `PYTHONPATH` is handled automatically by the venv.

Once the server is up, the existing React frontend (http://localhost:3000) can call:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/upload` | Upload CSV (wavenumber, intensity). Validates coverage (≥20 points, no NaN/duplicates), sorts/interpolates to WaveRef, then returns raw, baseline-corrected (imodpoly), and normalized spectra |
| `POST /api/preprocess` | Baseline correction / normalization (options: `none`, `baseline`, `normalization`, `both`) |
| `POST /api/denoise` | Membrane filter correction + simple Savitzky–Golay fallback denoise |
| `POST /api/classify` | Use correlation (`classification_model=disable`) or the requested AlexNet/LeNet5 variant. If a requested combo is missing, backend auto-falls back to correlation and returns a warning. Optionally send `baseline_intensities` to compare baseline vs denoised |
| `GET /api/models/info` | Lists available model files discovered under `backend/model` |

## 3. Classification details

- **Supported plastic types (22):** `["Acrylic", "Cellulose", ..., "PVC"]`
- **Spectrum length:** enforced at 1340 points (WaveRef grid).  
  Models expecting 1323 points are automatically resampled.
- **Model selection:**  
  - The frontend still posts `membrane_filter` (e.g., `"Cellulose Ester"`) and `denoising_model` (`"CAE"`, `"CNNAE-Xception"`, etc.).  
  - The backend maps these to the actual filenames (`ClassifierModel_{B|E}_20SNR_{CAE|Xception|...}_{AlexNet|LeNet5}.h5`).  
    - `"Cellulose Ester"` → `B`, `"Nylon"` → `E`.  
    - `"Glass Fiber"` currently maps to `D`, which is reserved (models not loaded yet) so the API will return 400 until those weights are added.
    - `"Disable"` → `NoDenoise`, `CNNAE-Xception` → `Xception`, `CNNAE-ResNet50` → `ResNet50`, `CNNAE-InceptionV3` → `InceptionV3`.
  - If an exact combination is missing the backend now logs it, exposes it via `missing_classifier_combos` in `GET /api/models/info`, and auto-falls back to correlation so the request still succeeds (with a warning string in the response).
- **Outputs:** plastic type, softmax confidence (%), gradient-based correlation (baseline vs denoised when `baseline_intensities` is provided), comparison spectrum (baseline or reference), model identifier, CAM-style heatmap (`cam_heatmap`) highlighting important wavenumber regions, and the top-3 reference matches.

## 4. Troubleshooting

| Issue | Fix |
| --- | --- |
| `HTTP 400 - CSV must contain at least …` | Input file has NaN/too few points or doesn’t cover 650–4000 cm⁻¹. Clean the CSV then re-upload |
| `ModuleNotFoundError: jax` | Activate the venv and rerun `pip install -r requirements.txt` |
| `RuntimeError: SynCleanSet.npy ...` | Ensure the reference dataset sits in `backend/SynCleanSet.npy` |
| `HTTP 400 - model not available` | Check `GET /api/models/info` for the exact membrane/denoise combinations that exist |
| Very slow first prediction | The first call compiles the JAX graph; subsequent requests are fast |

The backend is now ready for end-to-end testing with the redesigned UI.
