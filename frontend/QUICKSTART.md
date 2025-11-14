# Quick Start Guide - FTIR Analysis System v2.0

## 🚀 Get Started in 5 Minutes

### Step 1: Backend Setup (2 minutes)

```bash
# Navigate to project directory
cd ftir-redesigned

# Install Python dependencies
pip install fastapi uvicorn pandas numpy scipy --break-system-packages

# Optional: Install TensorFlow for actual model inference
pip install tensorflow --break-system-packages

# Start backend server
python main.py
```

Backend will be running at: **http://localhost:8000**

---

### Step 2: Frontend Setup (2 minutes)

```bash
# In a new terminal, navigate to project directory
cd ftir-redesigned

# Install Node.js dependencies
npm install

# Start development server
npm start
```

Frontend will open automatically at: **http://localhost:3000**

---

### Step 3: Test the Application (1 minute)

1. Click **"START ANALYSIS"** on landing page
2. Upload a CSV file (format: wavenumber, intensity)
3. Navigate through steps using NEXT button
4. Try the Apply/Clear workflow
5. See results in Step 4!

---

## 📁 Directory Structure After Setup

```
ftir-redesigned/
├── main.py                    ← Backend server
├── models/                    ← Place your .h5 files here
│   ├── step3/                 (12 denoising models)
│   └── step4/                 (30 classification models)
├── SynCleanSet.npy           ← Reference dataset (add this)
├── src/                       ← React frontend source
├── public/                    ← Static files
├── package.json               ← Node dependencies
└── README.md                  ← Full documentation
```

---

## 🎯 What to Do Next

### 1. Add Your Model Files

Create the model directories:
```bash
mkdir -p models/step3 models/step4
```

Add your .h5 model files using this naming convention:

**Step 3 (Denoising):**
```
models/step3/model_CE_CAE.h5
models/step3/model_CE_CNNAE-Xception.h5
models/step3/model_GF_CAE.h5
... (12 total)
```

**Step 4 (Classification):**
```
models/step4/classifier_CE_CAE_LeNet5.h5
models/step4/classifier_CE_CAE_AlexNet.h5
models/step4/classifier_GF_CNNAE-Xception_LeNet5.h5
... (30 total)
```

### 2. Add Reference Dataset

Place your `SynCleanSet.npy` file in the project root:
```bash
# Should contain: 220 samples × 1340 points
# Shape: (220, 1340)
```

### 3. Test with Real Data

Prepare a CSV file with your FTIR spectrum:
```csv
650.0,0.123
652.5,0.145
655.0,0.167
...
4000.0,0.089
```

---

## 🔧 Common Issues & Solutions

### Issue 1: Backend won't start
**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
pip install fastapi uvicorn pandas numpy scipy --break-system-packages
```

---

### Issue 2: Frontend won't start
**Error**: `Cannot find module 'react'`

**Solution**:
```bash
npm install
```

---

### Issue 3: CORS errors in browser
**Error**: `Access to fetch blocked by CORS policy`

**Solution**: Make sure backend is running on port 8000:
```bash
python main.py
```

---

### Issue 4: Model not found error
**Error**: `FileNotFoundError: model_CE_CAE.h5`

**Solution**: 
1. Create `models/step3/` and `models/step4/` directories
2. Add your .h5 model files
3. For testing without models, backend has placeholder code that will use simple filtering

---

## 📖 Key Features to Try

### 1. **Apply/Clear Workflow**
- Click Apply → button fades
- Click Clear (now enabled) → shows confirmation
- Clear resets, allowing you to Apply again

### 2. **Back Navigation**
- Use back button in top-right corner
- Go back to any previous step
- Charts persist when you return

### 3. **Step 4 Layout**
- Charts on the left
- Controls and results on the right
- Clean, organized interface

### 4. **Disable Options**
- Step 3: Disable = No denoising model
- Step 4: Disable = Correlation-based classification

---

## 🎨 Customization

### Change Colors
Edit `src/App.css`:
```css
:root {
  --primary-color: #YOUR_COLOR;
  --secondary-color: #YOUR_COLOR;
}
```

### Adjust Layout
Edit `src/App.css`:
```css
:root {
  --header-height: 70px;
  --sidebar-width: 300px;
}
```

---

## 📊 API Testing

### Test Upload
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_spectrum.csv"
```

### Test Classification
```bash
curl -X POST http://localhost:8000/api/classify \
  -F "intensities=[...]" \
  -F "membrane_filter=Cellulose Ester" \
  -F "denoising_model=CAE" \
  -F "classification_model=LeNet5"
```

### Get Model Info
```bash
curl http://localhost:8000/api/models/info
```

---

## 📱 Mobile Testing

To test on mobile device:

1. Find your computer's IP address:
```bash
# On Mac/Linux
ifconfig | grep inet

# On Windows
ipconfig
```

2. Update backend CORS if needed (already set to allow all)

3. Access from mobile browser:
```
http://YOUR_IP_ADDRESS:3000
```

---

## 🚀 Production Build

### Build Frontend
```bash
npm run build
```

Creates optimized production build in `build/` directory.

### Serve Production Build
```bash
# Using serve (install if needed)
npm install -g serve
serve -s build -p 3000
```

---

## 💡 Tips

1. **Start backend first**, then frontend
2. **Keep both terminals open** while developing
3. **Check console** for any errors
4. **Use Chrome DevTools** for debugging
5. **Test Apply/Clear workflow** thoroughly

---

## 📞 Support

If you encounter issues:

1. Check `CHANGES.md` for detailed implementation notes
2. Read `README.md` for comprehensive documentation
3. Review error messages in browser console
4. Check backend terminal for API errors

---

## ✅ Verification Checklist

Before deploying:

- [ ] Backend starts without errors
- [ ] Frontend starts and displays landing page
- [ ] Can upload CSV file in Step 1
- [ ] Can select and apply options in Step 2
- [ ] Can select MF and denoising in Step 3
- [ ] Can classify and see results in Step 4
- [ ] Back button works correctly
- [ ] Progress bar updates on Apply
- [ ] Clear shows confirmation modal
- [ ] Charts persist when navigating back

---

**Ready to go!** 🎉

Start with the landing page and work through each step. The application guides you through the entire analysis workflow.

For detailed information, see:
- **README.md** - Complete documentation
- **CHANGES.md** - Implementation details
- **API docs** - http://localhost:8000/docs (when backend is running)
