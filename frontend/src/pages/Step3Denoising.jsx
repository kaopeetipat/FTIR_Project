import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step3Denoising.css';

const MEMBRANE_FILTERS = [
  { value: 'Cellulose Ester', label: 'Cellulose Ester' },
  { value: 'Glass Fiber', label: 'Glass Fiber' },
  { value: 'Nylon', label: 'Nylon' }
];

const DENOISING_MODELS = [
  { value: 'disable', label: 'Disable (No Denoising)' },
  { value: 'CAE', label: 'CAE' },
  { value: 'CNNAE-Xception', label: 'CNNAE-Xception' },
  { value: 'CNNAE-ResNet50', label: 'CNNAE-ResNet50' },
  { value: 'CNNAE-InceptionV3', label: 'CNNAE-InceptionV3' }
];

function Step3Denoising({
  spectralData,
  setSpectralData,
  denoisingConfig,
  setDenoisingConfig,
  onNext,
  onApply,
  onClear,
  isApplied,
  canProceed,
  isMobile
}) {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [showClearModal, setShowClearModal] = useState(false);

  const handleMembraneFilterChange = (value) => {
    if (!isApplied) {
      setDenoisingConfig(prev => ({ ...prev, membraneFilter: value }));
    }
  };

  const handleDenoisingModelChange = (value) => {
    if (!isApplied) {
      setDenoisingConfig(prev => ({ ...prev, denoisingModel: value }));
    }
  };

  const handleApply = async () => {
    if (!denoisingConfig.membraneFilter || !denoisingConfig.denoisingModel || !canProceed) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      const inputIntensities = spectralData.preprocessedIntensities.length > 0 
        ? spectralData.preprocessedIntensities 
        : spectralData.originalIntensities;
      
      formData.append('intensities', JSON.stringify(inputIntensities));
      formData.append('membrane_filter', denoisingConfig.membraneFilter);
      formData.append('denoising_model', denoisingConfig.denoisingModel);

      const response = await fetch('http://localhost:8000/api/denoise', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to denoise spectrum');
      }

      const data = await response.json();
      
      setSpectralData(prev => ({
        ...prev,
        denoisedIntensities: data.denoisedSpectrum
      }));

      onApply();
    } catch (err) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const confirmClear = () => {
    setDenoisingConfig({ membraneFilter: null, denoisingModel: null });
    setSpectralData(prev => ({
      ...prev,
      denoisedIntensities: []
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  const handleNext = () => {
    if (isApplied) {
      onNext();
      navigate('/step4');
    }
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 2 first</h2>
            <p>You need to preprocess the spectrum before denoising.</p>
            <button className="next-button" onClick={() => navigate('/step2')}>
              Go to Step 2
            </button>
          </div>
        </div>
      </div>
    );
  }

  const isSelectionComplete = denoisingConfig.membraneFilter && denoisingConfig.denoisingModel;

  return (
    <div className="step-container">
      <div className="step-content">
        {/* Chart Panel */}
        <div className="chart-panel">
          <h2>Denoised Spectrum</h2>
          <div className="chart-wrapper">
            {spectralData.denoisedIntensities.length > 0 ? (
              <SpectrumChart 
                wavenumbers={spectralData.wavenumbers}
                preprocessedIntensities={
                  spectralData.preprocessedIntensities.length > 0 
                    ? spectralData.preprocessedIntensities 
                    : spectralData.originalIntensities
                }
                denoisedIntensities={spectralData.denoisedIntensities}
              />
            ) : (
              <div className="chart-placeholder">
                <p>Select membrane filter and denoising model, then click APPLY</p>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <h2>Denoising Configuration</h2>
          
          {/* Membrane Filter Selection */}
          <div className="radio-group">
            <h3>Membrane Filter</h3>
            {MEMBRANE_FILTERS.map(filter => (
              <label 
                key={filter.value}
                className={`radio-option ${denoisingConfig.membraneFilter === filter.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="membrane-filter"
                  value={filter.value}
                  checked={denoisingConfig.membraneFilter === filter.value}
                  onChange={(e) => handleMembraneFilterChange(e.target.value)}
                  disabled={isApplied}
                />
                <span className="radio-label">{filter.label}</span>
              </label>
            ))}
          </div>

          {/* Denoising Model Selection */}
          <div className="radio-group">
            <h3>Denoising Model</h3>
            {DENOISING_MODELS.map(model => (
              <label 
                key={model.value}
                className={`radio-option ${denoisingConfig.denoisingModel === model.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="denoising-model"
                  value={model.value}
                  checked={denoisingConfig.denoisingModel === model.value}
                  onChange={(e) => handleDenoisingModelChange(e.target.value)}
                  disabled={isApplied}
                />
                <span className="radio-label">{model.label}</span>
              </label>
            ))}
          </div>

          {error && <p className="error-text">{error}</p>}

          <div className="action-buttons">
            <button 
              className={`apply-button ${isApplied ? 'applied' : ''}`}
              onClick={handleApply}
              disabled={!isSelectionComplete || isApplied || isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="loading-spinner"></span>
                  Processing...
                </>
              ) : (
                'APPLY'
              )}
            </button>
            <button 
              className={`clear-button ${isApplied ? 'enabled' : ''}`}
              onClick={handleClearClick}
              disabled={!isApplied}
            >
              CLEAR
            </button>
          </div>

          <button 
            className="next-button"
            onClick={handleNext}
            disabled={!isApplied}
          >
            NEXT STEP
          </button>
        </div>
      </div>

      {/* Clear Confirmation Modal */}
      {showClearModal && (
        <div className="modal-overlay" onClick={() => setShowClearModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Confirm Clear</h3>
            <p>Are you sure you want to clear the denoising configuration? This will reset Step 4.</p>
            <div className="modal-buttons">
              <button className="modal-button confirm" onClick={confirmClear}>
                Yes, Clear
              </button>
              <button className="modal-button cancel" onClick={() => setShowClearModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Spectrum Chart Component with Comparison
function SpectrumChart({ wavenumbers, preprocessedIntensities, denoisedIntensities }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = Math.min(...wavenumbers);
  const xMax = Math.max(...wavenumbers);
  
  const allIntensities = [...preprocessedIntensities, ...denoisedIntensities];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) => 
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  
  const yScale = (y) => 
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const preprocessedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(preprocessedIntensities[i])}`)
    .join(' ');

  const denoisedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(denoisedIntensities[i])}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      <defs>
        <linearGradient id="denoisedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#059669" />
          <stop offset="100%" stopColor="#10b981" />
        </linearGradient>
      </defs>

      {/* Grid */}
      <g className="grid" opacity="0.1">
        {[0, 1, 2, 3, 4].map(i => {
          const y = padding.top + (i * (height - padding.top - padding.bottom) / 4);
          return (
            <line 
              key={`h-${i}`}
              x1={padding.left} 
              y1={y} 
              x2={width - padding.right} 
              y2={y}
              stroke="#666"
              strokeWidth="1"
            />
          );
        })}
      </g>

      {/* Preprocessed Spectrum (faded) */}
      <path 
        d={preprocessedPath}
        fill="none"
        stroke="#666"
        strokeWidth="1"
        opacity="0.3"
        strokeDasharray="3,3"
      />

      {/* Denoised Spectrum */}
      <path 
        d={denoisedPath}
        fill="none"
        stroke="url(#denoisedGradient)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Axes */}
      <line 
        x1={padding.left} 
        y1={height - padding.bottom}
        x2={width - padding.right}
        y2={height - padding.bottom}
        stroke="#666"
        strokeWidth="2"
      />
      <line 
        x1={padding.left}
        y1={padding.top}
        x2={padding.left}
        y2={height - padding.bottom}
        stroke="#666"
        strokeWidth="2"
      />

      {/* Legend */}
      <g transform="translate(460, 30)">
        <line x1="0" y1="0" x2="30" y2="0" stroke="#666" strokeWidth="1" strokeDasharray="3,3" opacity="0.3"/>
        <text x="35" y="5" fill="#999" fontSize="12">Input</text>
        
        <line x1="0" y1="15" x2="30" y2="15" stroke="url(#denoisedGradient)" strokeWidth="2"/>
        <text x="35" y="20" fill="#999" fontSize="12">Denoised</text>
      </g>

      {/* Labels */}
      <text x={width / 2} y={height - 10} textAnchor="middle" fill="#999" fontSize="14">
        Wavenumber (cm⁻¹)
      </text>
      <text x={20} y={height / 2} textAnchor="middle" fill="#999" fontSize="14"
        transform={`rotate(-90, 20, ${height / 2})`}>
        Intensity
      </text>
    </svg>
  );
}

export default Step3Denoising;
