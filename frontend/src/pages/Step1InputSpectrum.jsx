import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step1InputSpectrum.css';

function Step1InputSpectrum({ 
  uploadedFile, 
  setUploadedFile, 
  spectralData, 
  setSpectralData,
  onNext,
  onApply,
  onClear,
  isApplied,
  isMobile 
}) {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showClearModal, setShowClearModal] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }, []);

  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFile = async (file) => {
    if (!file.name.endsWith('.csv')) {
      setError('Only CSV files are accepted');
      return;
    }

    setError(null);
    setIsProcessing(true);
    setUploadedFile(file);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      const data = await response.json();
      
      setSpectralData(prev => ({
        ...prev,
        wavenumbers: data.wavenumbers,
        originalIntensities: data.intensities,
        baselineIntensities: data.baseline_corrected || [],
        normalizedIntensities: data.normalized_intensities || []
      }));

      onApply();
      setIsProcessing(false);
    } catch (err) {
      setError(err.message);
      setIsProcessing(false);
      setUploadedFile(null);
    }
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const confirmClear = () => {
    setUploadedFile(null);
    setSpectralData(prev => ({
      ...prev,
      wavenumbers: [],
      originalIntensities: [],
      baselineIntensities: [],
      normalizedIntensities: []
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  const handleNext = () => {
    if (isApplied) {
      onNext();
      navigate('/step2');
    }
  };

  return (
    <div className="step-container">
      <div className="step-content">
        {/* Chart Panel */}
        <div className="chart-panel">
          <h2>Input Spectrum</h2>
          <div className="chart-wrapper">
            {spectralData.originalIntensities.length > 0 ? (
              <SpectrumChart 
                wavenumbers={spectralData.wavenumbers}
                intensities={spectralData.originalIntensities}
              />
            ) : (
              <div className="chart-placeholder">
                <p>Upload a CSV file to view the spectrum</p>
              </div>
            )}
          </div>
        </div>

        {/* Control Panel */}
        <div className="control-panel">
          <h2>File Upload</h2>
          
          {!uploadedFile ? (
            <div 
              className={`upload-area ${isDragging ? 'drag-active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input').click()}
            >
              <input
                id="file-input"
                type="file"
                accept=".csv"
                onChange={handleFileInput}
                style={{ display: 'none' }}
              />
              <div className="upload-icon">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none">
                  <path d="M7 18C4.79086 18 3 16.2091 3 14C3 11.7909 4.79086 10 7 10C7 7.23858 9.23858 5 12 5C14.7614 5 17 7.23858 17 10C19.2091 10 21 11.7909 21 14C21 16.2091 19.2091 18 17 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M12 12V19M12 12L9 15M12 12L15 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <p className="upload-text">
                {isDragging ? 'Drop file here' : 'Drag & drop CSV file here'}
              </p>
              <p className="upload-subtext">or click to browse</p>
            </div>
          ) : (
            <div className="file-info">
              <div className="file-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
                  <path d="M13 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V9L13 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  <path d="M13 2V9H20" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              <div className="file-details">
                <p className="file-name">{uploadedFile.name}</p>
                <p className="file-size">
                  {(uploadedFile.size / 1024).toFixed(2)} KB
                </p>
                {spectralData.originalIntensities.length > 0 && (
                  <p className="file-points">
                    {spectralData.originalIntensities.length} data points
                  </p>
                )}
              </div>
            </div>
          )}

          {error && (
            <p className="error-text">{error}</p>
          )}

          {isProcessing && (
            <div className="processing-indicator">
              <div className="loading-spinner"></div>
              <span>Processing file...</span>
            </div>
          )}

          <div className="action-buttons">
            <button 
              className={`clear-button ${uploadedFile ? 'enabled' : ''}`}
              onClick={handleClearClick}
              disabled={!uploadedFile}
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
            <p>Are you sure you want to clear the uploaded file? This will reset all subsequent steps.</p>
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

// Spectrum Chart Component
function SpectrumChart({ wavenumbers, intensities }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = Math.min(...wavenumbers);
  const xMax = Math.max(...wavenumbers);
  const yMin = Math.min(...intensities);
  const yMax = Math.max(...intensities);

  const xScale = (x) => 
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  
  const yScale = (y) => 
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const pathData = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(intensities[i])}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      <defs>
        <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
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

      {/* Spectrum Line */}
      <path 
        d={pathData}
        fill="none"
        stroke="url(#lineGradient)"
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

      {/* Labels */}
      <text 
        x={width / 2} 
        y={height - 10}
        textAnchor="middle"
        fill="#999"
        fontSize="14"
      >
        Wavenumber (cm⁻¹)
      </text>
      <text 
        x={20} 
        y={height / 2}
        textAnchor="middle"
        fill="#999"
        fontSize="14"
        transform={`rotate(-90, 20, ${height / 2})`}
      >
        Intensity
      </text>

      {/* Tick Labels */}
      <text x={padding.left} y={height - padding.bottom + 20} textAnchor="middle" fill="#999" fontSize="12">
        {xMin.toFixed(0)}
      </text>
      <text x={width - padding.right} y={height - padding.bottom + 20} textAnchor="middle" fill="#999" fontSize="12">
        {xMax.toFixed(0)}
      </text>
    </svg>
  );
}

export default Step1InputSpectrum;
