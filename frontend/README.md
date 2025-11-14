# FTIR Microplastic Analysis System v2.0

Deep Learning Denoising and Classification for Enhanced Microplastic FTIR Identification

## 🎯 Overview

This is a redesigned full-stack application for FTIR microplastic analysis with:
- **Frontend**: React.js with improved UX/UI
- **Backend**: FastAPI with comprehensive model management
- **Features**: Apply/Clear workflow, confirmation modals, vertical centering, and optimized navigation

## 🚀 Key Features

### Frontend Improvements

#### 1. **Improved Navigation**
- Back button relocated to top-right of header
- Navigate to previous step with single click
- Progress updates automatically when "Apply" is clicked at each stage
- Charts persist when navigating back through steps

#### 2. **Apply/Clear Workflow**
- **Apply Button**: Click once to process, then button fades (can't click again)
- **Clear Button**: Default faded state, becomes active after Apply is clicked
- **Confirmation Modal**: All clear actions require user confirmation to prevent accidental data loss
- Must click Clear before you can Apply again

#### 3. **Layout Improvements**
- **Vertical Centering**: All components stay vertically centered in main area
- **No Bottom Scrolling**: Page content fits within viewport (no scroll down on desktop)
- **Step 4 Layout**: Separate chart panel (left) and control panel (right)
- **Responsive**: Mobile-friendly with stacked layouts

#### 4. **Model Configuration**
- **Step 3**: 
  - Choose Membrane Filter (Cellulose Ester, Glass Fiber, Nylon)
  - Choose Denoising Model (Disable, CAE, CNNAE-Xception, CNNAE-ResNet50, CNNAE-InceptionV3)
  - "Disable" means no denoising model is applied
- **Step 4**:
  - Choose Classification (Disable, LeNet5, AlexNet)
  - "Disable" uses correlation-based classification (from main.py)

### Backend Architecture

#### 1. **Model Structure**
Based on senior template (`main.py` with 696 lines), the backend manages:

**WaveRef Configuration**:
```python
WaveRef = np.arange(650, 4000, 2.5)  # 1340 points
```

**Step 3 - Denoising Models** (12 total):
- 3 Membrane Filters × 4 Denoising Models = 12 models
- Format: `model_{MF}_{DenoiseModel}.h5`
- Example: `model_CE_CAE.h5`, `model_GF_CNNAE-Xception.h5`

**Step 4 - Classification Models** (30 total):
- 3 Membrane Filters × 4 Denoising Models × 2.5 Classification Models = 30 models
- Format: `classifier_{MF}_{DenoiseModel}_{ClassificationModel}.h5`
- Example: `classifier_CE_CAE_LeNet5.h5`

#### 2. **API Endpoints**

```
POST /api/upload          - Upload CSV spectrum file
POST /api/preprocess      - Apply baseline correction and/or normalization
POST /api/denoise         - Apply membrane filter correction and denoising
POST /api/classify        - Classify spectrum using DL or correlation
GET  /api/models/info     - Get available models and configuration info
```

#### 3. **Processing Pipeline**

```
Step 1: Upload CSV → Interpolate to WaveRef → Store original intensities
   ↓
Step 2: Apply preprocessing → Baseline correction and/or normalization
   ↓
Step 3: Apply membrane filter + Denoising model → Get denoised spectrum
   ↓
Step 4: Apply classification → Get plastic type, accuracy, correlation
```

## 📁 Project Structure

```
ftir-redesigned/
├── main.py                         # Backend FastAPI server
├── models/                         # Model files directory
│   ├── step3/                      # Denoising models
│   │   ├── model_CE_CAE.h5
│   │   ├── model_CE_CNNAE-Xception.h5
│   │   ├── ...
│   └── step4/                      # Classification models
│       ├── classifier_CE_CAE_LeNet5.h5
│       ├── classifier_CE_CAE_AlexNet.h5
│       ├── ...
├── src/
│   ├── App.jsx                     # Main app with routing
│   ├── App.css                     # Global styles
│   ├── index.js                    # Entry point
│   ├── index.css                   # Base styles
│   ├── components/
│   │   ├── Header.jsx              # Top navigation with back button
│   │   ├── Header.css
│   │   ├── Sidebar.jsx             # Progress tracking sidebar
│   │   └── Sidebar.css
│   └── pages/
│       ├── LandingPage.jsx         # Welcome screen
│       ├── LandingPage.css
│       ├── Step1InputSpectrum.jsx  # File upload
│       ├── Step1InputSpectrum.css
│       ├── Step2Preprocessing.jsx  # Preprocessing options
│       ├── Step2Preprocessing.css
│       ├── Step3Denoising.jsx      # MF + Denoising selection
│       ├── Step3Denoising.css
│       ├── Step4Classification.jsx # Classification results
│       └── Step4Classification.css
├── public/
│   └── index.html
├── package.json
└── README.md
```

## 🛠️ Installation & Setup

### Backend Setup

1. **Install Python dependencies**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn pandas numpy scipy tensorflow pybaselines

#exit
deactivate
```

2. **Prepare model files**:
```bash
# Create model directories
mkdir -p models/step3 models/step4

# Place your .h5 model files in respective directories
# Step 3: models/step3/model_{MF}_{DenoiseModel}.h5
# Step 4: models/step4/classifier_{MF}_{DenoiseModel}_{ClassModel}.h5
```

3. **Prepare reference dataset**:
```bash
# Place SynCleanSet.npy in the same directory as main.py
# This file should contain clean reference spectra (220 samples × 1340 points)
```

4. **Start backend server**:
```bash
python main.py
# Server runs on http://localhost:8000
```

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
cd ftir-redesigned
npm install
```

2. **Start development server**:
```bash
npm start
# Frontend runs on http://localhost:3000
```

3. **Build for production**:
```bash
npm run build
# Creates optimized build in build/ directory
```

## 🎨 Design Specifications

### Color Scheme
```css
--primary-color: #7B2CBF      (Purple)
--secondary-color: #C77DFF    (Light Purple)
--dark-bg: #1a1a1a           (Near Black)
--success-color: #059669      (Green)
--error-color: #dc2626        (Red)
```

### Layout Dimensions
- **Header Height**: 70px
- **Sidebar Width**: 300px (desktop)
- **Desktop**: 1920×1080 - No scrolling, all components fit
- **Mobile**: Responsive with vertical scrolling

## 📊 User Workflow

### Step 1: Input Spectrum
1. Upload CSV file (drag & drop or click to browse)
2. File is automatically processed and chart displays
3. Click NEXT to proceed to Step 2

### Step 2: Preprocessing
1. Select preprocessing option (none, baseline, normalization, or both)
2. Click APPLY to process
3. View comparison chart (original vs processed)
4. Click NEXT to proceed to Step 3

### Step 3: Denoising
1. Select Membrane Filter
2. Select Denoising Model (or Disable for no denoising)
3. Click APPLY to process
4. View comparison chart (input vs denoised)
5. Click NEXT to proceed to Step 4

### Step 4: Classification
1. Select Classification Model (or Disable for correlation-based)
2. Click APPLY to classify
3. View results: Plastic Type, Accuracy, Correlation
4. View comparison chart (input vs reference)
5. Analysis complete!

## 🔧 API Usage Examples

### Upload Spectrum
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@spectrum.csv"
```

### Preprocess
```bash
curl -X POST http://localhost:8000/api/preprocess \
  -F "intensities=[0.1,0.2,...]" \
  -F "preprocessing_option=both"
```

### Denoise
```bash
curl -X POST http://localhost:8000/api/denoise \
  -F "intensities=[0.1,0.2,...]" \
  -F "membrane_filter=Cellulose Ester" \
  -F "denoising_model=CAE"
```

### Classify
```bash
curl -X POST http://localhost:8000/api/classify \
  -F "intensities=[0.1,0.2,...]" \
  -F "membrane_filter=Cellulose Ester" \
  -F "denoising_model=CAE" \
  -F "classification_model=LeNet5"
```

## 🎯 Key Improvements from v1.0

1. ✅ **Relocated back button** to header top-right
2. ✅ **Progress updates** on Apply click (not on navigation)
3. ✅ **Chart persistence** when navigating back
4. ✅ **Vertical centering** of all components
5. ✅ **No bottom scrolling** on desktop
6. ✅ **Step 4 left-right layout** for charts and controls
7. ✅ **Apply/Clear workflow** with fading buttons
8. ✅ **Confirmation modals** for all clear actions
9. ✅ **Comprehensive backend** with model management
10. ✅ **Correlation-based classification** for "Disable" mode

## 📝 Model File Naming Convention

### Step 3 (Denoising Models)

| Membrane Filter | Denoising Model | Filename |
|----------------|-----------------|----------|
| Cellulose Ester | CAE | `model_CE_CAE.h5` |
| Cellulose Ester | CNNAE-Xception | `model_CE_CNNAE-Xception.h5` |
| Glass Fiber | CAE | `model_GF_CAE.h5` |
| Nylon | CNNAE-ResNet50 | `model_NY_CNNAE-ResNet50.h5` |

### Step 4 (Classification Models)

| MF | Denoise Model | Classification | Filename |
|----|---------------|----------------|----------|
| CE | CAE | LeNet5 | `classifier_CE_CAE_LeNet5.h5` |
| CE | CAE | AlexNet | `classifier_CE_CAE_AlexNet.h5` |
| GF | CNNAE-Xception | LeNet5 | `classifier_GF_CNNAE-Xception_LeNet5.h5` |

## 🔍 Troubleshooting

### Backend Issues

**Issue**: Model not found
```
Solution: Check that model files are in correct directories with correct naming
```

**Issue**: SynCleanSet.npy not found
```
Solution: Place reference dataset in same directory as main.py
```

**Issue**: CORS errors
```
Solution: Backend has CORS enabled for all origins. Check if backend is running.
```

### Frontend Issues

**Issue**: API connection failed
```
Solution: Ensure backend is running on http://localhost:8000
```

**Issue**: Charts not displaying
```
Solution: Check browser console for errors. Verify CSV format is correct.
```

**Issue**: Clear button not working
```
Solution: You must click Apply first before Clear becomes active
```

## 👥 Contributors

- Chatchanan Khamtonwong (6522771029)
- Puntawat Rattananuntakorn (6522772472)
- Anas Langu (6522771946)
- Peetipat Sakontarat (6522772399)

**Advisor**: Seksan Laitrakun

## 📄 License

© 2025 SIIT - Thammasat University

---

**Version**: 2.0.0  
**Last Updated**: November 2025  
**Status**: Production Ready
