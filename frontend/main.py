from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import interpolate
from scipy.signal import savgol_filter
import logging
from io import StringIO
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Define constants ---
# WaveRef as specified by user
WaveRef = np.arange(650, 4000, 2.5)
TARGET_SPEC_LENGTH = len(WaveRef)  # 1340 points

logger.info(f"WaveRef initialized with {TARGET_SPEC_LENGTH} points from 650 to 4000 cm⁻¹")

# --- Define materials and sample counts ---
NameList = ["Acrylic", "Cellulose", "ENR", "EPDM", "HDPE", "LDPE", "Nylon", "PBAT", "PBS", "PC",
            "PEEK", "PEI", "PET", "PLA", "PMMA", "POM", "PP", "PS", "PTFE", "PU", "PVA", "PVC"]
NumCleanSpec = 10  # Number of clean spectra per material

# --- Load SynCleanSet dataset ---
# This should be updated to match the new WaveRef length
try:
    # For now, create a synthetic dataset
    # In production, load from file: SynCleanSet = np.load("SynCleanSet.npy")
    total_samples = len(NameList) * NumCleanSpec
    SynCleanSet = np.random.rand(total_samples, TARGET_SPEC_LENGTH)
    logger.info(f"Successfully initialized SynCleanSet with shape {SynCleanSet.shape}")
except Exception as e:
    logger.error(f"Error loading SynCleanSet: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading reference dataset: {str(e)}")

# --- Model Configuration ---
# Step 3: Membrane Filter options (3 types) × Denoising Models (4 types) = 12 models
MEMBRANE_FILTERS = ["Cellulose Ester", "Glass Fiber", "Nylon"]
DENOISING_MODELS = ["CAE", "CNNAE-Xception", "CNNAE-ResNet50", "CNNAE-InceptionV3"]

# Step 4: Classification Models
CLASSIFICATION_MODELS = ["LeNet5", "AlexNet"]

# Model path structure
# Format: model_{MF}_{DenoiseModel}.h5
# Example: model_CE_CAE.h5, model_GF_CNNAE-Xception.h5
def get_denoising_model_path(membrane_filter, denoising_model):
    """Generate model path for Step 3 denoising"""
    mf_code = {
        "Cellulose Ester": "CE",
        "Glass Fiber": "GF",
        "Nylon": "NY"
    }[membrane_filter]
    
    model_name = f"model_{mf_code}_{denoising_model}.h5"
    return os.path.join("models", "step3", model_name)

# Format: classifier_{MF}_{DenoiseModel}_{ClassificationModel}.h5
# Example: classifier_CE_CAE_LeNet5.h5
def get_classification_model_path(membrane_filter, denoising_model, classification_model):
    """Generate model path for Step 4 classification"""
    mf_code = {
        "Cellulose Ester": "CE",
        "Glass Fiber": "GF",
        "Nylon": "NY"
    }[membrane_filter]
    
    model_name = f"classifier_{mf_code}_{denoising_model}_{classification_model}.h5"
    return os.path.join("models", "step4", model_name)

# Placeholder for loaded models
loaded_models = {
    "step3": {},  # Will store denoising models
    "step4": {}   # Will store classification models
}

def load_denoising_model(membrane_filter, denoising_model):
    """Load or retrieve cached denoising model"""
    model_key = f"{membrane_filter}_{denoising_model}"
    
    if model_key not in loaded_models["step3"]:
        model_path = get_denoising_model_path(membrane_filter, denoising_model)
        logger.info(f"Loading denoising model: {model_path}")
        # In production, load actual model:
        # from tensorflow.keras.models import load_model
        # loaded_models["step3"][model_key] = load_model(model_path, compile=False)
        loaded_models["step3"][model_key] = None  # Placeholder
    
    return loaded_models["step3"][model_key]

def load_classification_model(membrane_filter, denoising_model, classification_model):
    """Load or retrieve cached classification model"""
    model_key = f"{membrane_filter}_{denoising_model}_{classification_model}"
    
    if model_key not in loaded_models["step4"]:
        model_path = get_classification_model_path(membrane_filter, denoising_model, classification_model)
        logger.info(f"Loading classification model: {model_path}")
        # In production, load actual model:
        # from tensorflow.keras.models import load_model
        # loaded_models["step4"][model_key] = load_model(model_path, compile=False)
        loaded_models["step4"][model_key] = None  # Placeholder
    
    return loaded_models["step4"][model_key]

# --- Helper Functions ---
def interpolate_spectrum(wavenumbers, intensities, target_wavenumbers):
    """Interpolate spectrum to match target wavenumbers"""
    f = interpolate.interp1d(wavenumbers, intensities, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    return f(target_wavenumbers)

def baseline_correction(spectrum):
    """Apply baseline correction using asymmetric least squares"""
    try:
        from pybaselines import Baseline
        baseline_fitter = Baseline(x_data=np.arange(len(spectrum)))
        baseline = baseline_fitter.asls(spectrum, lam=1e6, p=0.01)[0]
        return spectrum - baseline
    except:
        # Fallback: simple polynomial baseline
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, 3)
        baseline = np.polyval(coeffs, x)
        return spectrum - baseline

def minmax_normalization(spectrum):
    """Apply Min-Max normalization"""
    min_val = np.min(spectrum)
    max_val = np.max(spectrum)
    if max_val - min_val == 0:
        return spectrum
    return (spectrum - min_val) / (max_val - min_val)

def apply_membrane_filter_correction(spectrum, membrane_filter):
    """Apply membrane filter specific corrections"""
    # Placeholder for filter-specific processing
    # Different filters may have different spectral artifacts
    logger.info(f"Applying membrane filter correction: {membrane_filter}")
    
    # In production, implement actual filter correction algorithms
    # For now, return as-is
    return spectrum

def apply_denoising(spectrum, model):
    """Apply denoising model to spectrum"""
    # Placeholder for actual model inference
    # In production:
    # spectrum_reshaped = spectrum.reshape(1, -1, 1)
    # denoised = model.predict(spectrum_reshaped)
    # return denoised.flatten()
    
    # For now, apply simple Savitzky-Golay filter as placeholder
    return savgol_filter(spectrum, window_length=11, polyorder=3)

def classify_by_correlation(spectrum, reference_set):
    """Classify spectrum using Pearson correlation (for 'Disable' mode)"""
    best_score = -1
    best_class = None
    best_spectrum = None
    
    for i, material_name in enumerate(NameList):
        start_idx = i * NumCleanSpec
        end_idx = (i + 1) * NumCleanSpec
        
        if start_idx >= len(reference_set):
            break
            
        material_spectra = reference_set[start_idx:end_idx]
        avg_spectrum = np.mean(material_spectra, axis=0)
        
        if np.std(spectrum) == 0 or np.std(avg_spectrum) == 0:
            score = 0.0
        else:
            score, _ = pearsonr(spectrum, avg_spectrum)
            if np.isnan(score):
                score = 0.0
        
        if score > best_score:
            best_score = score
            best_class = material_name
            best_spectrum = avg_spectrum
    
    return best_class, best_score, best_spectrum

def classify_with_model(spectrum, model):
    """Classify spectrum using deep learning model"""
    # Placeholder for actual model inference
    # In production:
    # spectrum_reshaped = spectrum.reshape(1, -1, 1)
    # predictions = model.predict(spectrum_reshaped)
    # class_idx = np.argmax(predictions)
    # confidence = predictions[0][class_idx]
    # return NameList[class_idx], confidence
    
    # For now, use correlation as fallback
    return classify_by_correlation(spectrum, SynCleanSet)[:2]

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "FTIR Microplastic Analysis Backend", "version": "2.0"}

@app.post("/api/upload")
async def upload_spectrum(file: UploadFile = File(...)):
    """
    Upload and parse CSV file
    Expected format: wavenumber, intensity
    """
    try:
        logger.info(f"Received file upload: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')), header=None)
        
        if df.shape[1] < 2:
            raise HTTPException(status_code=400, detail="CSV must have at least 2 columns")
        
        wavenumbers = df.iloc[:, 0].values.astype(float)
        intensities = df.iloc[:, 1].values.astype(float)
        
        # Interpolate to standard WaveRef
        interpolated_intensities = interpolate_spectrum(wavenumbers, intensities, WaveRef)
        
        logger.info(f"Successfully processed spectrum: {len(interpolated_intensities)} points")
        
        return JSONResponse(content={
            'wavenumbers': WaveRef.tolist(),
            'intensities': interpolated_intensities.tolist(),
            'original_length': len(wavenumbers),
            'processed_length': len(interpolated_intensities)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preprocess")
async def preprocess_spectrum(
    intensities: str = Form(...),
    preprocessing_option: str = Form(...)
):
    """
    Apply preprocessing to spectrum
    Options: 'none', 'baseline', 'normalization', 'both'
    """
    try:
        logger.info(f"Preprocessing request: {preprocessing_option}")
        
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)
        
        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(status_code=400, 
                detail=f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, got {len(spectrum)}")
        
        processed = spectrum.copy()
        
        if preprocessing_option in ['baseline', 'both']:
            processed = baseline_correction(processed)
            logger.info("Applied baseline correction")
        
        if preprocessing_option in ['normalization', 'both']:
            processed = minmax_normalization(processed)
            logger.info("Applied normalization")
        
        return JSONResponse(content={
            'processedSpectrum': processed.tolist(),
            'preprocessing_applied': preprocessing_option
        })
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/denoise")
async def denoise_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form(...),
    denoising_model: str = Form(...)
):
    """
    Apply denoising to spectrum
    membrane_filter: 'Cellulose Ester', 'Glass Fiber', 'Nylon'
    denoising_model: 'disable', 'CAE', 'CNNAE-Xception', 'CNNAE-ResNet50', 'CNNAE-InceptionV3'
    """
    try:
        logger.info(f"Denoising request: MF={membrane_filter}, Model={denoising_model}")
        
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)
        
        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(status_code=400, 
                detail=f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, got {len(spectrum)}")
        
        # Apply membrane filter correction
        processed = apply_membrane_filter_correction(spectrum, membrane_filter)
        
        # Apply denoising model
        if denoising_model.lower() != 'disable':
            model = load_denoising_model(membrane_filter, denoising_model)
            if model is not None:
                processed = apply_denoising(processed, model)
            else:
                # Fallback if model not loaded
                processed = apply_denoising(processed, None)
            logger.info(f"Applied denoising model: {denoising_model}")
        else:
            logger.info("Denoising disabled - no model applied")
        
        # Ensure output is normalized
        processed = minmax_normalization(processed)
        
        return JSONResponse(content={
            'denoisedSpectrum': processed.tolist(),
            'membrane_filter': membrane_filter,
            'denoising_model': denoising_model
        })
        
    except Exception as e:
        logger.error(f"Denoising error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify")
async def classify_spectrum(
    intensities: str = Form(...),
    membrane_filter: str = Form(...),
    denoising_model: str = Form(...),
    classification_model: str = Form(...)
):
    """
    Classify spectrum
    classification_model: 'disable', 'LeNet5', 'AlexNet'
    'disable' uses correlation-based classification
    """
    try:
        logger.info(f"Classification request: Model={classification_model}")
        
        intensities_list = json.loads(intensities)
        spectrum = np.array(intensities_list, dtype=np.float32)
        
        if len(spectrum) != TARGET_SPEC_LENGTH:
            raise HTTPException(status_code=400, 
                detail=f"Spectrum length mismatch. Expected {TARGET_SPEC_LENGTH}, got {len(spectrum)}")
        
        if classification_model.lower() == 'disable':
            # Use correlation-based classification
            plastic_type, correlation, clean_spectrum = classify_by_correlation(spectrum, SynCleanSet)
            accuracy = correlation * 100  # Convert to percentage
            
            logger.info(f"Correlation classification: {plastic_type}, {correlation:.4f}")
            
            return JSONResponse(content={
                'plastic_type': plastic_type,
                'accuracy': float(accuracy),
                'correlation': float(correlation),
                'clean_spectrum': clean_spectrum.tolist() if clean_spectrum is not None else [],
                'method': 'correlation'
            })
        else:
            # Use deep learning classification
            model = load_classification_model(membrane_filter, denoising_model, classification_model)
            plastic_type, confidence = classify_with_model(spectrum, model)
            
            # Get reference spectrum for comparison
            material_idx = NameList.index(plastic_type)
            start_idx = material_idx * NumCleanSpec
            end_idx = (material_idx + 1) * NumCleanSpec
            clean_spectrum = np.mean(SynCleanSet[start_idx:end_idx], axis=0)
            
            # Calculate correlation for additional metric
            correlation, _ = pearsonr(spectrum, clean_spectrum)
            
            logger.info(f"DL classification: {plastic_type}, {confidence:.4f}")
            
            return JSONResponse(content={
                'plastic_type': plastic_type,
                'accuracy': float(confidence * 100),
                'correlation': float(correlation),
                'clean_spectrum': clean_spectrum.tolist(),
                'method': classification_model
            })
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/info")
async def get_models_info():
    """Get information about available models"""
    return JSONResponse(content={
        'membrane_filters': MEMBRANE_FILTERS,
        'denoising_models': ['Disable'] + DENOISING_MODELS,
        'classification_models': ['Disable'] + CLASSIFICATION_MODELS,
        'total_step3_models': len(MEMBRANE_FILTERS) * len(DENOISING_MODELS),
        'total_step4_models': len(MEMBRANE_FILTERS) * len(DENOISING_MODELS) * len(CLASSIFICATION_MODELS),
        'materials': NameList,
        'waveref_range': [float(WaveRef[0]), float(WaveRef[-1])],
        'waveref_points': TARGET_SPEC_LENGTH
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
