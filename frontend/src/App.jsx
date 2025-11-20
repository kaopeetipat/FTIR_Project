import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import LandingPage from './pages/LandingPage';
import Step1InputSpectrum from './pages/Step1InputSpectrum';
import Step2Preprocessing from './pages/Step2Preprocessing';
import Step3Denoising from './pages/Step3Denoising';
import Step4Classification from './pages/Step4Classification';
import './App.css';

function App() {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const [currentStep, setCurrentStep] = useState(0);
  const [progressPercentage, setProgressPercentage] = useState(0);
  
  // Application state - shared across all steps
  const [spectralData, setSpectralData] = useState({
    wavenumbers: [],
    originalIntensities: [],
    baselineIntensities: [],
    normalizedIntensities: [],
    preprocessedIntensities: [],
    denoisedIntensities: [],
    classificationResult: null
  });
  
  // Step-specific states
  const [uploadedFile, setUploadedFile] = useState(null);
  const [preprocessingOption, setPreprocessingOption] = useState(null);
  const [denoisingConfig, setDenoisingConfig] = useState({
    membraneFilter: null,
    denoisingModel: null
  });
  const [classificationModel, setClassificationModel] = useState(null);
  
  // Track which steps have been applied
  const [appliedSteps, setAppliedSteps] = useState({
    step1: false,
    step2: false,
    step3: false,
    step4: false
  });

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Update progress based on applied steps
  useEffect(() => {
    const completedSteps = Object.values(appliedSteps).filter(Boolean).length;
    setProgressPercentage((completedSteps / 4) * 100);
  }, [appliedSteps]);

  const handleStepChange = (step) => {
    setCurrentStep(step);
  };

  const markStepAsApplied = (step) => {
    setAppliedSteps(prev => ({
      ...prev,
      [step]: true
    }));
  };

  const resetStepData = (step) => {
    setAppliedSteps(prev => ({
      ...prev,
      [step]: false
    }));
    
    // Reset subsequent steps when clearing
    if (step === 'step1') {
      setSpectralData({
        wavenumbers: [],
        originalIntensities: [],
        baselineIntensities: [],
        normalizedIntensities: [],
        preprocessedIntensities: [],
        denoisedIntensities: [],
        classificationResult: null
      });
      setUploadedFile(null);
      setPreprocessingOption(null);
      setDenoisingConfig({ membraneFilter: null, denoisingModel: null });
      setClassificationModel(null);
      setAppliedSteps({
        step1: false,
        step2: false,
        step3: false,
        step4: false
      });
    } else if (step === 'step2') {
      setSpectralData(prev => ({
        ...prev,
        baselineIntensities: prev.baselineIntensities,
        normalizedIntensities: prev.normalizedIntensities,
        preprocessedIntensities: [],
        denoisedIntensities: [],
        classificationResult: null
      }));
      setPreprocessingOption(null);
      setDenoisingConfig({ membraneFilter: null, denoisingModel: null });
      setClassificationModel(null);
      setAppliedSteps(prev => ({
        ...prev,
        step2: false,
        step3: false,
        step4: false
      }));
    } else if (step === 'step3') {
      setSpectralData(prev => ({
        ...prev,
        denoisedIntensities: [],
        classificationResult: null
      }));
      setDenoisingConfig({ membraneFilter: null, denoisingModel: null });
      setClassificationModel(null);
      setAppliedSteps(prev => ({
        ...prev,
        step3: false,
        step4: false
      }));
    } else if (step === 'step4') {
      setSpectralData(prev => ({
        ...prev,
        classificationResult: null
      }));
      setClassificationModel(null);
      setAppliedSteps(prev => ({
        ...prev,
        step4: false
      }));
    }
  };

  return (
    <Router>
      <div className="app">
        <Header 
          isMobile={isMobile}
          currentStep={currentStep}
          onStepChange={handleStepChange}
          appliedSteps={appliedSteps}
        />
        
        <Sidebar 
          currentStep={currentStep}
          progressPercentage={progressPercentage}
          onStepClick={handleStepChange}
          isMobile={isMobile}
          appliedSteps={appliedSteps}
        />

        <main className="main-content">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route 
              path="/step1" 
              element={
                <Step1InputSpectrum 
                  uploadedFile={uploadedFile}
                  setUploadedFile={setUploadedFile}
                  spectralData={spectralData}
                  setSpectralData={setSpectralData}
                  onNext={() => handleStepChange(1)}
                  onApply={() => markStepAsApplied('step1')}
                  onClear={() => resetStepData('step1')}
                  isApplied={appliedSteps.step1}
                  isMobile={isMobile}
                />
              } 
            />
            <Route 
              path="/step2" 
              element={
                <Step2Preprocessing 
                  uploadedFile={uploadedFile}
                  spectralData={spectralData}
                  setSpectralData={setSpectralData}
                  preprocessingOption={preprocessingOption}
                  setPreprocessingOption={setPreprocessingOption}
                  onNext={() => handleStepChange(2)}
                  onApply={() => markStepAsApplied('step2')}
                  onClear={() => resetStepData('step2')}
                  isApplied={appliedSteps.step2}
                  canProceed={appliedSteps.step1}
                  isMobile={isMobile}
                />
              } 
            />
            <Route 
              path="/step3" 
              element={
                <Step3Denoising 
                  spectralData={spectralData}
                  setSpectralData={setSpectralData}
                  denoisingConfig={denoisingConfig}
                  setDenoisingConfig={setDenoisingConfig}
                  onNext={() => handleStepChange(3)}
                  onApply={() => markStepAsApplied('step3')}
                  onClear={() => resetStepData('step3')}
                  isApplied={appliedSteps.step3}
                  canProceed={appliedSteps.step2}
                  isMobile={isMobile}
                />
              } 
            />
            <Route
              path="/step4"
              element={
                <Step4Classification
                  uploadedFile={uploadedFile}
                  spectralData={spectralData}
                  setSpectralData={setSpectralData}
                  denoisingConfig={denoisingConfig}
                  classificationModel={classificationModel}
                  setClassificationModel={setClassificationModel}
                  onApply={() => markStepAsApplied('step4')}
                  onClear={() => resetStepData('step4')}
                  isApplied={appliedSteps.step4}
                  canProceed={appliedSteps.step3}
                  isMobile={isMobile}
                />
              }
            />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
