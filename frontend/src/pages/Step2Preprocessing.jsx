import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step2Preprocessing.css';

const PREPROCESSING_OPTIONS = [
  { value: 'none', label: 'No Processing' },
  { value: 'baseline', label: 'Baseline Correction Only' },
  { value: 'normalization', label: 'Min-Max Normalization Only' },
  { value: 'both', label: 'Baseline Correction and Min-Max Normalization' }
];

function Step2Preprocessing({
  spectralData,
  setSpectralData,
  preprocessingOption,
  setPreprocessingOption,
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

  const handleApply = async () => {
    if (!preprocessingOption || !canProceed) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('intensities', JSON.stringify(spectralData.originalIntensities));
      formData.append('preprocessing_option', preprocessingOption);

      const response = await fetch('http://localhost:8000/api/preprocess', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to preprocess spectrum');
      }

      const data = await response.json();
      
      setSpectralData(prev => ({
        ...prev,
        preprocessedIntensities: data.processedSpectrum
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
    setPreprocessingOption(null);
    setSpectralData(prev => ({
      ...prev,
      preprocessedIntensities: []
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  const handleNext = () => {
    if (isApplied) {
      onNext();
      navigate('/step3');
    }
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 1 first</h2>
            <p>You need to upload a spectrum file before preprocessing.</p>
            <button className="next-button" onClick={() => navigate('/step1')}>
              Go to Step 1
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="step-container">
      <div className="step-content">
        {/* Chart Panel */}
        <div className="chart-panel">
          <h2>Preprocessed Spectrum</h2>
          <div className="chart-wrapper">
            {spectralData.preprocessedIntensities.length > 0 ? (
              <SpectrumChart 
                wavenumbers={spectralData.wavenumbers}
                originalIntensities={spectralData.originalIntensities}
                processedIntensities={spectralData.preprocessedIntensities}
              />
            ) : (
              <div className="chart-placeholder">
                <p>Select a preprocessing option and click APPLY</p>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <h2>Preprocessing Options</h2>
          
          <div className="radio-group">
            {PREPROCESSING_OPTIONS.map(option => (
              <label 
                key={option.value}
                className={`radio-option ${preprocessingOption === option.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="preprocessing"
                  value={option.value}
                  checked={preprocessingOption === option.value}
                  onChange={(e) => setPreprocessingOption(e.target.value)}
                  disabled={isApplied}
                />
                <span className="radio-label">{option.label}</span>
              </label>
            ))}
          </div>

          {error && <p className="error-text">{error}</p>}

          <div className="action-buttons">
            <button 
              className={`apply-button ${isApplied ? 'applied' : ''}`}
              onClick={handleApply}
              disabled={!preprocessingOption || isApplied || isProcessing}
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
            <p>Are you sure you want to clear the preprocessing? This will reset all subsequent steps.</p>
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
function SpectrumChart({ wavenumbers, originalIntensities, processedIntensities }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = Math.min(...wavenumbers);
  const xMax = Math.max(...wavenumbers);
  
  const allIntensities = [...originalIntensities, ...processedIntensities];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) => 
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  
  const yScale = (y) => 
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const originalPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(originalIntensities[i])}`)
    .join(' ');

  const processedPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(processedIntensities[i])}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      <defs>
        <linearGradient id="processedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#7B2CBF" />
          <stop offset="100%" stopColor="#C77DFF" />
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

      {/* Original Spectrum (faded) */}
      <path 
        d={originalPath}
        fill="none"
        stroke="#666"
        strokeWidth="1"
        opacity="0.3"
        strokeDasharray="3,3"
      />

      {/* Processed Spectrum */}
      <path 
        d={processedPath}
        fill="none"
        stroke="url(#processedGradient)"
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
        <text x="35" y="5" fill="#999" fontSize="12">Original</text>
        
        <line x1="0" y1="15" x2="30" y2="15" stroke="url(#processedGradient)" strokeWidth="2"/>
        <text x="35" y="20" fill="#999" fontSize="12">Processed</text>
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

export default Step2Preprocessing;
