"""
FastAPI backend for FTIR microplastic classification.

This service exposes the same endpoints that the redesigned frontend expects
but upgrades Step 4 so that it can execute the real AlexNet/LeNet5 classifiers
that live in backend/model/*.h5.  Models were exported with Keras 3, so we run
them with the JAX backend (KERAS_BACKEND=jax) inside backend/venv.
"""

import os
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import logging

import keras
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Logging / app bootstrap
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ftir-backend")

app = FastAPI(title="FTIR Microplastic Analysis Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants & reference data
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SYN_CLEAN_PATH = BASE_DIR / "SynCleanSet.npy"

WaveRef = np.arange(650, 4000, 2.5)
TARGET_SPEC_LENGTH = len(WaveRef)  # 1340 points required by the user

MIN_UPLOAD_POINTS = 20
MIN_WAVENUMBER = 600.0
MAX_WAVENUMBER = 4100.0
MIN_DYNAMIC_RANGE = 1e-8  # minimum required std ratio

NameList = [
    "Acrylic",
    "Cellulose",
    "ENR",
    "EPDM",
    "HDPE",
    "LDPE",
    "Nylon",
    "PBAT",
    "PBS",
    "PC",
    "PEEK",
    "PEI",
    "PET",
    "PLA",
    "PMMA",
    "POM",
    "PP",
    "PS",
    "PTFE",
    "PU",
    "PVA",
    "PVC",
]

if not SYN_CLEAN_PATH.exists():
    raise RuntimeError(f"Reference dataset not found at {SYN_CLEAN_PATH}")

SynCleanSet = np.load(SYN_CLEAN_PATH)
if SynCleanSet.shape[1] != TARGET_SPEC_LENGTH:
    raise RuntimeError(
        f"SynCleanSet.npy second dimension must be {TARGET_SPEC_LENGTH}, "
        f"got {SynCleanSet.shape[1]}"
    )

samples_per_material = SynCleanSet.shape[0] // len(NameList)
if samples_per_material == 0:
    raise RuntimeError("SynCleanSet does not contain samples for all materials")

REFERENCE_MEANS: Dict[str, np.ndarray] = {}
for idx, material in enumerate(NameList):
    start = idx * samples_per_material
    end = start + samples_per_material
    REFERENCE_MEANS[material] = np.mean(SynCleanSet[start:end], axis=0)

MEMBRANE_FILTERS = ["Cellulose Ester", "Glass Fiber", "Nylon"]
DENOISING_MODELS = ["CAE", "CNNAE-Xception", "CNNAE-ResNet50", "CNNAE-InceptionV3"]
CLASSIFICATION_MODELS = ["LeNet5", "AlexNet"]

MEMBRANE_CODE_ALIAS = {
    "cellulose ester": "B",
    "glass fiber": "D",
    "nylon": "E",
    "b": "B",
    "e": "E",
    "d": "D",
}

DENOISING_NAME_ALIAS = {
    "disable": "NoDenoise",
    "none": "NoDenoise",
    "cae": "CAE",
    "cnnae-xception": "Xception",
    "cnnae-resnet50": "ResNet50",
    "cnnae-inceptionv3": "InceptionV3",
    "xception": "Xception",
    "resnet50": "ResNet50",
    "inceptionv3": "InceptionV3",
    "nodenoise": "NoDenoise",
}

EXPECTED_MEMBRANE_CODES = ["B", "D", "E"]
EXPECTED_CLASSIFIER_DENOISERS = ["CAE", "Xception", "ResNet50", "InceptionV3", "NoDenoise"]

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

class ModelManager:
    """Discovers and loads AlexNet / LeNet5 classifiers on demand."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.registry = self._discover_models()
        self.cache: Dict[str, Tuple[keras.Model, int]] = {}
        self.missing: List[Tuple[str, str, str]] = self._validate_inventory()

        if not self.registry:
            raise RuntimeError(f"No classifier .h5 files found in {model_dir}")

    def _discover_models(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        registry: Dict[str, Dict[str, Dict[str, Path]]] = {}
        for path in self.model_dir.glob("ClassifierModel_*.h5"):
            parts = path.stem.split("_")
            if len(parts) != 5 or parts[0] != "ClassifierModel":
                continue

            _, membrane_code, snr_tag, denoise_name, classifier_name = parts
            if classifier_name not in CLASSIFICATION_MODELS:
                continue

            registry.setdefault(classifier_name, {}).setdefault(membrane_code, {})[
                denoise_name
            ] = path

        logger.info("Discovered classifier variants: %s", registry)
        return registry

    def describe(self) -> Dict[str, Dict[str, List[str]]]:
        """Return nested dict for documentation."""
        summary: Dict[str, Dict[str, List[str]]] = {}
        for classifier, membranes in self.registry.items():
            summary[classifier] = {
                membrane: sorted(denoisers.keys()) for membrane, denoisers in membranes.items()
            }
        return summary

    def _validate_inventory(self) -> List[Tuple[str, str, str]]:
        missing: List[Tuple[str, str, str]] = []
        for classifier in CLASSIFICATION_MODELS:
            classifier_dict = self.registry.get(classifier, {})
            for membrane in EXPECTED_MEMBRANE_CODES:
                membrane_dict = classifier_dict.get(membrane, {})
                for denoiser in EXPECTED_CLASSIFIER_DENOISERS:
                    if denoiser not in membrane_dict:
                        missing.append((classifier, membrane, denoiser))

        if missing:
            logger.warning(
                "Missing classifier model files: %s",
                [f"{c}/{m}/{d}" for c, m, d in missing],
            )
        else:
            logger.info("All expected classifier combinations are available.")
        return missing

    def resolve_identifier(
        self, classifier: str, membrane_code: str, denoiser: str
    ) -> Path:
        classifier_dict = self.registry.get(classifier)
        if not classifier_dict:
            raise HTTPException(
                status_code=400,
                detail=f"Classifier '{classifier}' not available. "
                f"Available: {list(self.registry.keys())}",
            )

        membrane_dict = classifier_dict.get(membrane_code)
        if not membrane_dict:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Membrane code '{membrane_code}' not available for classifier '{classifier}'. "
                    f"Available codes: {list(classifier_dict.keys())}"
                ),
            )

        model_path = membrane_dict.get(denoiser)
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising variant '{denoiser}' not available for classifier '{classifier}' "
                    f"with membrane '{membrane_code}'. "
                    f"Choices: {list(membrane_dict.keys())}"
                ),
            )
        return model_path

    def load(self, model_path: Path) -> Tuple[keras.Model, int]:
        key = model_path.stem
        if key not in self.cache:
            logger.info("Loading classifier %s", model_path.name)
            model = keras.models.load_model(model_path, compile=False)
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            required_len = int(input_shape[1])
            self.cache[key] = (model, required_len)
        return self.cache[key]


model_manager = ModelManager(MODEL_DIR)


class DenoiseModelManager:
    """Loads membrane-filter specific denoising autoencoders."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.registry = self._discover_models()
        self.cache: Dict[str, Tuple[keras.Model, int, int]] = {}

        if not self.registry:
            raise RuntimeError(f"No TrainingModel_*.h5 files found in {model_dir}")

    def _discover_models(self) -> Dict[str, Dict[str, Path]]:
        registry: Dict[str, Dict[str, Path]] = {}
        for path in self.model_dir.glob("TrainingModel_*.h5"):
            parts = path.stem.split("_")
            if len(parts) != 4 or parts[0] != "TrainingModel":
                continue

            _, membrane_code, _snr_tag, denoise_name = parts
            registry.setdefault(membrane_code, {})[denoise_name] = path

        logger.info("Discovered denoising models: %s", registry)
        return registry

    def describe(self) -> Dict[str, List[str]]:
        return {
            membrane: sorted(denoisers.keys())
            for membrane, denoisers in self.registry.items()
        }

    def resolve_identifier(self, membrane_code: str, denoiser: str) -> Path:
        membrane_dict = self.registry.get(membrane_code)
        if not membrane_dict:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising models for membrane '{membrane_code}' not available. "
                    f"Available membranes: {list(self.registry.keys())}"
                ),
            )

        model_path = membrane_dict.get(denoiser)
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Denoising model '{denoiser}' not available for membrane '{membrane_code}'. "
                    f"Choices: {list(membrane_dict.keys())}"
                ),
            )
        return model_path

    def load(self, model_path: Path) -> Tuple[keras.Model, int, int]:
        key = model_path.stem
        if key not in self.cache:
            logger.info("Loading denoising model %s", model_path.name)
            model = keras.models.load_model(model_path, compile=False)

            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            output_shape = model.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]

            input_len = int(input_shape[1])
            output_len = int(output_shape[1])
            self.cache[key] = (model, input_len, output_len)

        return self.cache[key]


denoise_manager = DenoiseModelManager(MODEL_DIR)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def interpolate_spectrum(wavenumbers, intensities, target_wavenumbers):
    """Interpolate spectrum to match the standard WaveRef grid."""
    f = interpolate.interp1d(
        wavenumbers,
        intensities,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f(target_wavenumbers)


def baseline_correction(spectrum, poly_order: int = 6):
    """Apply baseline correction using pybaselines imodpoly, with fallbacks."""
    try:
        from pybaselines import polynomial

        baseline = polynomial.imodpoly(spectrum, poly_order=poly_order)[0]
        return spectrum - baseline
    except Exception:
        try:
            from pybaselines import Baseline

            baseline_fitter = Baseline(x_data=np.arange(len(spectrum)))
            baseline = baseline_fitter.asls(spectrum, lam=1e6, p=0.01)[0]
            return spectrum - baseline
        except Exception:  # pragma: no cover - fallback for environments without pybaselines
            x = np.arange(len(spectrum))
            coeffs = np.polyfit(x, spectrum, 3)
            return spectrum - np.polyval(coeffs, x)


def minmax_normalization(spectrum: np.ndarray) -> np.ndarray:
    min_val = np.min(spectrum)
    max_val = np.max(spectrum)
    if np.isclose(max_val - min_val, 0):
        return spectrum
    return (spectrum - min_val) / (max_val - min_val)


def process_uploaded_spectrum(
    wavenumbers: np.ndarray, intensities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce senior pipeline: interpolate -> baseline -> normalize."""
    interpolated = interpolate_spectrum(wavenumbers, intensities, WaveRef)
    baseline_corrected = baseline_correction(interpolated)
    normalized = minmax_normalization(baseline_corrected)
    return interpolated, baseline_corrected, normalized


def apply_membrane_filter_correction(spectrum, membrane_filter):
    logger.debug("Applying membrane filter correction: %s", membrane_filter)
    return spectrum


def apply_fallback_denoising(spectrum):
    """Fallback denoising (Savitzky-Golay) when DL models are not available."""
    return savgol_filter(spectrum, window_length=11, polyorder=3)


def run_denoising_model(
    spectrum: np.ndarray, model_tuple: Tuple[keras.Model, int, int]
) -> np.ndarray:
    model, input_len, output_len = model_tuple
    prepared = prepare_for_model(spectrum, input_len)
    prediction = model.predict(prepared, verbose=0)
    denoised = np.asarray(prediction).reshape(output_len)
    denoised = resample_signal(denoised, TARGET_SPEC_LENGTH)
    return denoised


def resample_signal(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D signal to the requested length."""
    if len(signal) == target_len:
        return signal
    x_src = np.linspace(0.0, 1.0, len(signal))
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, signal)


def prepare_for_model(signal: np.ndarray, required_len: int) -> np.ndarray:
    processed = resample_signal(signal, required_len)
    processed = processed.reshape(1, required_len, 1).astype("float32")
    return processed


def classify_by_correlation(spectrum: np.ndarray):
    best_material = None
    best_score = -1.0
    best_reference = None

    for material, reference in REFERENCE_MEANS.items():
        if np.std(reference) == 0 or np.std(spectrum) == 0:
            score = 0.0
        else:
            score, _ = pearsonr(spectrum, reference)
            if np.isnan(score):
                score = 0.0
        if score > best_score:
            best_material = material
            best_score = score
            best_reference = reference

    return best_material, float(best_score), best_reference


def correlation_rankings(spectrum: np.ndarray, top_k: int = 3):
    scores = []
    for material, reference in REFERENCE_MEANS.items():
        if np.std(reference) == 0 or np.std(spectrum) == 0:
            score = 0.0
        else:
            score, _ = pearsonr(spectrum, reference)
            if np.isnan(score):
                score = 0.0
        scores.append((material, float(score)))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:top_k]


def safe_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value, _ = pearsonr(a, b)
    if np.isnan(value):
        return 0.0
    return float(value)


def gradient_correlation(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Correlation between gradients (RowCorr style)."""
    if a is None or b is None:
        return 0.0

    if len(a) != len(b):
        b = resample_signal(b, len(a))

    grad_a = np.gradient(a)
    grad_b = np.gradient(b)

    # Ensure 2D for DataFrame correlation
    grad_a_2d = grad_a if grad_a.ndim > 1 else grad_a.reshape(-1, 1)
    grad_b_2d = grad_b if grad_b.ndim > 1 else grad_b.reshape(-1, 1)

    df_a = pd.DataFrame(grad_a_2d)
    df_b = pd.DataFrame(grad_b_2d)

    row_corr = df_a.corrwith(df_b, axis=1)
    row_corr = row_corr.replace([np.inf, -np.inf], np.nan).dropna()

    if row_corr.empty:
        return safe_correlation(grad_a, grad_b)

    return float(row_corr.mean())


def generate_cam_heatmap(
    model: keras.Model,
    prepared_input: np.ndarray,
    class_idx: int,
    window_size: int = 64,
    stride: int = 32,
) -> np.ndarray:
    """Occlusion-based class activation approximation along the spectrum."""
    window_size = max(8, min(window_size, prepared_input.shape[1]))
    stride = max(4, stride)

    base_pred = model.predict(prepared_input, verbose=0)[0][class_idx]
    input_len = prepared_input.shape[1]
    heatmap = np.zeros(input_len, dtype=np.float32)

    for start in range(0, input_len, stride):
        end = min(input_len, start + window_size)
        perturbed = prepared_input.copy()
        perturbed[0, start:end, 0] = 0.0
        pred = model.predict(perturbed, verbose=0)[0][class_idx]
        drop = max(base_pred - pred, 0.0)
        heatmap[start:end] += drop

    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap


def classify_with_model(
    spectrum: np.ndarray, model: keras.Model, required_len: int
) -> Tuple[str, float, List[float], np.ndarray, int]:
    prepared = prepare_for_model(spectrum, required_len)
    predictions = model.predict(prepared, verbose=0)[0]
    class_idx = int(np.argmax(predictions))
    return (
        NameList[class_idx],
        float(predictions[class_idx]),
        predictions.tolist(),
        prepared,
        class_idx,
    )


def resolve_membrane_code(value: str) -> str:
    code = value.strip()
    if not code:
        return "B"
    normalized = code.lower()
    return MEMBRANE_CODE_ALIAS.get(normalized, "B")


def resolve_denoising_name(value: str) -> str:
    normalized = value.strip().lower()
    return DENOISING_NAME_ALIAS.get(normalized, value)


def sanitize_uploaded_spectrum(
    wavenumbers: np.ndarray, intensities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate, clean, and sort incoming spectra before interpolation."""
    if len(wavenumbers) != len(intensities):
        raise HTTPException(
            status_code=400, detail="Wavenumber and intensity column lengths differ"
        )

    mask = ~(np.isnan(wavenumbers) | np.isnan(intensities))
    wavenumbers = wavenumbers[mask]
    intensities = intensities[mask]

    if len(wavenumbers) < MIN_UPLOAD_POINTS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain at least {MIN_UPLOAD_POINTS} valid rows",
        )

    order = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[order]
    intensities = intensities[order]

    wavenumbers, unique_idx = np.unique(wavenumbers, return_index=True)
    intensities = intensities[unique_idx]

    low, high = wavenumbers[0], wavenumbers[-1]
    if low > MIN_WAVENUMBER or high < MAX_WAVENUMBER:
        logger.warning(
            "Partial wavenumber coverage detected (%.1f–%.1f). Results may be extrapolated.",
            low,
            high,
        )

    if np.std(intensities) < MIN_DYNAMIC_RANGE:
        raise HTTPException(
            status_code=400,
            detail="Intensity range too small; check baseline/normalization",
        )

    return wavenumbers, intensities


def ensure_finite(label: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise HTTPException(
            status_code=400,
            detail=f"{label} contains NaN or infinite values. Check preprocessing pipeline.",
        )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"message": "FTIR Microplastic Analysis Backend", "version": "2.0"}


@app.post("/api/upload")
async def upload_spectrum(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")

        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), header=None)

        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400, detail="CSV must have at least 2 columns"
            )

        wavenumbers = df.iloc[:, 0].values.astype(float)
        intensities = df.iloc[:, 1].values.astype(float)

        wavenumbers, intensities = sanitize_uploaded_spectrum(
            wavenumbers, intensities
        )

        (
            interpolated_intensities,
            baseline_corrected,
            normalized_intensities,
        ) = process_uploaded_spectrum(wavenumbers, intensities)

        return JSONResponse(
            content={
                "wavenumbers": WaveRef.tolist(),
                "intensities": interpolated_intensities.tolist(),
                "baseline_corrected": baseline_corrected.tolist(),
                "normalized_intensities": normalized_intensities.tolist(),
                "original_length": len(wavenumbers),
                "processed_length": len(interpolated_intensities),
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network/file errors
        logger.exception("Upload error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/preprocess")
async def preprocess_spectrum(
    intensities: str = Form(...),
    preprocessing_option: str = Form(...),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)
        ensure_finite("Classification spectrum", spectrum)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        processed = spectrum.copy()
        option = preprocessing_option.lower()

        if option in {"baseline", "both"}:
            processed = baseline_correction(processed)
            ensure_finite("Baseline-corrected spectrum", processed)

        if option in {"normalization", "both"}:
            processed = minmax_normalization(processed)
            ensure_finite("Normalized spectrum", processed)

        return JSONResponse(
            content={
                "processedSpectrum": processed.tolist(),
                "preprocessing_applied": preprocessing_option,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Preprocessing error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/denoise")
async def denoise_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form(...),
    denoising_model: str = Form(...),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        processed = apply_membrane_filter_correction(spectrum, membrane_filter)

        if denoising_model.lower() != "disable":
            membrane_code = resolve_membrane_code(membrane_filter)
            denoise_name = resolve_denoising_name(denoising_model)
            model_path = denoise_manager.resolve_identifier(membrane_code, denoise_name)
            try:
                processed = run_denoising_model(
                    processed, denoise_manager.load(model_path)
                )
                logger.info(
                    "Applied denoising model %s (%s)", denoise_name, model_path.name
                )
            except Exception:  # pragma: no cover
                logger.exception("Denoising model failed; applying fallback")
                processed = apply_fallback_denoising(processed)
        else:
            logger.info("Denoising disabled - no model applied")

        processed = minmax_normalization(processed)
        ensure_finite("Denoised spectrum", processed)

        return JSONResponse(
            content={
                "denoisedSpectrum": processed.tolist(),
                "membrane_filter": membrane_filter,
                "denoising_model": denoising_model,
            }
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Denoising error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/classify")
async def classify_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form(...),
    denoising_model: str = Form(...),
    classification_model: str = Form(...),
    baseline_intensities: Optional[str] = Form(None),
):
    try:
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)

        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, "
                    f"got {len(spectrum)}"
                ),
            )

        baseline_spectrum: Optional[np.ndarray] = None
        if baseline_intensities:
            baseline_list = json.loads(baseline_intensities)
            baseline_array = np.array(baseline_list, dtype=np.float32)
            if len(baseline_array) != TARGET_SPEC_LENGTH:
                baseline_array = resample_signal(baseline_array, TARGET_SPEC_LENGTH)
            ensure_finite("Baseline spectrum", baseline_array)
            baseline_spectrum = baseline_array

        if classification_model.lower() == "disable":
            plastic_type, correlation, clean_spectrum = classify_by_correlation(
                spectrum
            )
            comparison_spectrum = (
                baseline_spectrum if baseline_spectrum is not None else clean_spectrum
            )
            correlation_value = (
                gradient_correlation(baseline_spectrum, spectrum)
                if baseline_spectrum is not None
                else correlation
            )

            return JSONResponse(
                content={
                    "plastic_type": plastic_type,
                    "accuracy": float(max(correlation, 0) * 100.0),
                    "correlation": float(correlation_value),
                    "clean_spectrum": comparison_spectrum.tolist()
                    if comparison_spectrum is not None
                    else [],
                    "reference_spectrum": clean_spectrum.tolist()
                    if clean_spectrum is not None
                    else [],
                    "baseline_spectrum": baseline_spectrum.tolist()
                    if baseline_spectrum is not None
                    else [],
                    "method": "correlation",
                    "cam_heatmap": [],
                    "top_matches": [
                        {"plastic_type": name, "correlation": score}
                        for name, score in correlation_rankings(spectrum, top_k=3)
                    ],
                }
            )

        membrane_code = resolve_membrane_code(membrane_filter)
        denoise_name = resolve_denoising_name(denoising_model)

        model_path = None
        try:
            model_path = model_manager.resolve_identifier(
                classification_model, membrane_code, denoise_name
            )
            model, required_len = model_manager.load(model_path)
        except HTTPException as model_error:
            logger.warning(
                "Classifier combo unavailable (%s/%s/%s): %s; falling back to correlation.",
                membrane_code,
                denoise_name,
                classification_model,
                model_error.detail,
            )
            plastic_type, correlation, clean_spectrum = classify_by_correlation(
                spectrum
            )
            comparison_spectrum = (
                baseline_spectrum if baseline_spectrum is not None else clean_spectrum
            )
            correlation_value = (
                gradient_correlation(baseline_spectrum, spectrum)
                if baseline_spectrum is not None
                else correlation
            )
            return JSONResponse(
                content={
                    "plastic_type": plastic_type,
                    "accuracy": float(max(correlation, 0) * 100.0),
                    "correlation": float(correlation_value),
                    "clean_spectrum": comparison_spectrum.tolist()
                    if comparison_spectrum is not None
                    else [],
                    "reference_spectrum": clean_spectrum.tolist()
                    if clean_spectrum is not None
                    else [],
                    "baseline_spectrum": baseline_spectrum.tolist()
                    if baseline_spectrum is not None
                    else [],
                    "method": "correlation_fallback",
                    "warning": model_error.detail,
                    "cam_heatmap": [],
                    "top_matches": [
                        {"plastic_type": name, "correlation": score}
                        for name, score in correlation_rankings(spectrum, top_k=3)
                    ],
                }
            )

        (
            plastic_type,
            confidence,
            probabilities,
            prepared_input,
            class_idx,
        ) = classify_with_model(spectrum, model, required_len)

        clean_reference = REFERENCE_MEANS.get(plastic_type)
        comparison_spectrum = (
            baseline_spectrum if baseline_spectrum is not None else clean_reference
        )
        correlation_value = (
            gradient_correlation(baseline_spectrum, spectrum)
            if baseline_spectrum is not None
            else gradient_correlation(clean_reference, spectrum)
        )

        cam_heatmap = []
        try:
            raw_cam = generate_cam_heatmap(
                model, prepared_input, class_idx, window_size=64, stride=32
            )
            cam_heatmap = resample_signal(raw_cam, TARGET_SPEC_LENGTH).tolist()
        except Exception:  # pragma: no cover
            logger.exception("Failed to generate CAM heatmap")
            cam_heatmap = []

        response_payload = {
            "plastic_type": plastic_type,
            "accuracy": float(confidence * 100.0),
            "correlation": float(correlation_value),
            "clean_spectrum": comparison_spectrum.tolist()
            if comparison_spectrum is not None
            else [],
            "reference_spectrum": clean_reference.tolist()
            if clean_reference is not None
            else [],
            "baseline_spectrum": baseline_spectrum.tolist()
            if baseline_spectrum is not None
            else [],
            "method": classification_model,
            "model_identifier": model_path.stem,
            "probabilities": probabilities,
            "cam_heatmap": cam_heatmap,
            "top_matches": [
                {"plastic_type": name, "correlation": score}
                for name, score in correlation_rankings(spectrum, top_k=3)
            ],
        }
        return JSONResponse(content=response_payload)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Classification error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models/info")
async def get_models_info():
    return JSONResponse(
        content={
            "membrane_filters": MEMBRANE_FILTERS,
            "denoising_models": ["Disable"] + DENOISING_MODELS,
            "classification_models": ["Disable"] + CLASSIFICATION_MODELS,
            "materials": NameList,
            "waveref_range": [float(WaveRef[0]), float(WaveRef[-1])],
            "waveref_points": TARGET_SPEC_LENGTH,
            "samples_per_material": samples_per_material,
            "available_denoise_models": denoise_manager.describe(),
            "available_classifier_models": model_manager.describe(),
            "missing_classifier_combos": [
                {
                    "classification_model": classifier,
                    "membrane_code": membrane,
                    "denoising_variant": denoiser,
                }
                for classifier, membrane, denoiser in model_manager.missing
            ],
        }
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
