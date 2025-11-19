import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step4Classification.css';

const CLASSIFICATION_MODELS = [
  { value: 'disable', label: <>Correlation <br/> (Library Search)</> },
  { value: 'LeNet5', label: 'LeNet5' },
  { value: 'AlexNet', label: 'AlexNet' }
];

function Step4Classification({
  spectralData,
  setSpectralData,
  denoisingConfig,
  classificationModel,
  setClassificationModel,
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
  const [results, setResults] = useState(null);
  const [activeVisualization, setActiveVisualization] = useState('spectrum');
  const [showPreprocessed, setShowPreprocessed] = useState(true);
  const [showDenoised, setShowDenoised] = useState(true);
  const [showClassification, setShowClassification] = useState(true);

  const handleApply = async () => {
    if (!classificationModel || !canProceed) return;

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      const inputIntensities = spectralData.denoisedIntensities.length > 0 
        ? spectralData.denoisedIntensities 
        : (spectralData.preprocessedIntensities.length > 0 
          ? spectralData.preprocessedIntensities 
          : spectralData.originalIntensities);
      
      formData.append('intensities', JSON.stringify(inputIntensities));
      formData.append('membrane_filter', denoisingConfig.membraneFilter || 'Cellulose Ester');
      formData.append('denoising_model', denoisingConfig.denoisingModel || 'disable');
      formData.append('classification_model', classificationModel);
      const baselineSource =
        spectralData.preprocessedIntensities.length > 0
          ? spectralData.preprocessedIntensities
          : (spectralData.baselineIntensities && spectralData.baselineIntensities.length > 0)
            ? spectralData.baselineIntensities
            : spectralData.originalIntensities;

      if (baselineSource && baselineSource.length > 0) {
        formData.append('baseline_intensities', JSON.stringify(baselineSource));
      }

      const response = await fetch('http://localhost:8000/api/classify', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to classify spectrum');
      }

      const data = await response.json();
      
      setResults({
        plasticType: data.plastic_type,
        correlation: data.correlation,
        cleanSpectrum: data.clean_spectrum,
        classificationReference: data.reference_spectrum || [],
        warning: data.warning || null,
        camHeatmap: data.cam_heatmap || []
      });

      setSpectralData(prev => ({
        ...prev,
        classificationResult: data
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
    setClassificationModel(null);
    setResults(null);
    setSpectralData(prev => ({
      ...prev,
      classificationResult: null
    }));
    setError(null);
    onClear();
    setShowClearModal(false);
  };

  if (!canProceed) {
    return (
      <div className="step-container">
        <div className="step-content single-column">
          <div className="info-message">
            <h2>Please complete Step 3 first</h2>
            <p>You need to denoise the spectrum before classification.</p>
            <button className="next-button" onClick={() => navigate('/step3')}>
              Go to Step 3
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="step-container step4">
      <div className="step-content step4-layout">
        {/* Left Side - Charts */}
        <div className="chart-panel-container">
          {/* Classification Model Selection (Top of Left Panel) */}
          <div className="model-selection-card">
            <h3>Classification Model</h3>
            <div className="radio-group-inline">
              {CLASSIFICATION_MODELS.map(model => (
                <label 
                  key={model.value}
                  className={`radio-option-inline ${classificationModel === model.value ? 'selected' : ''}`}
                >
                  <input
                    type="radio"
                    name="classification-model"
                    value={model.value}
                    checked={classificationModel === model.value}
                    onChange={(e) => setClassificationModel(e.target.value)}
                    disabled={isApplied}
                  />
                  <span className="radio-label">{model.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Visualization Tabs */}
          <div className="chart-panel">
            <div className="visualization-tabs">
              <button
                className={`viz-tab ${activeVisualization === 'spectrum' ? 'active' : ''}`}
                onClick={() => setActiveVisualization('spectrum')}
              >
                Spectrum Comparison
              </button>
              <button
                className={`viz-tab ${activeVisualization === 'cam' ? 'active' : ''}`}
                onClick={() => setActiveVisualization('cam')}
                disabled={!results || !results.camHeatmap || results.camHeatmap.length === 0}
                title={!results || !results.camHeatmap || results.camHeatmap.length === 0 ? 'Run AlexNet/LeNet5 to view CAM' : ''}
              >
                Class Activation Map
              </button>
            </div>
            <div className="chart-wrapper">
              {results ? (
                activeVisualization === 'spectrum' ? (
                  <ComparisonChart
                    wavenumbers={spectralData.wavenumbers}
                    preprocessedIntensities={
                      spectralData.preprocessedIntensities.length > 0
                        ? spectralData.preprocessedIntensities
                        : spectralData.originalIntensities
                    }
                    denoisedIntensities={
                      spectralData.denoisedIntensities.length > 0
                        ? spectralData.denoisedIntensities
                        : spectralData.preprocessedIntensities
                    }
                    classificationReference={
                      results.classificationReference?.length
                        ? results.classificationReference
                        : results.cleanSpectrum
                    }
                    camHeatmap={results.camHeatmap}
                    showHeatmap={false}
                    showPreprocessed={showPreprocessed}
                    showDenoised={showDenoised}
                    showClassification={showClassification}
                  />
                ) : (
                  <CamChart
                    wavenumbers={spectralData.wavenumbers}
                    inputIntensities={
                      spectralData.denoisedIntensities.length > 0
                        ? spectralData.denoisedIntensities
                        : (spectralData.preprocessedIntensities.length > 0
                          ? spectralData.preprocessedIntensities
                          : spectralData.originalIntensities)
                    }
                    camHeatmap={results.camHeatmap}
                  />
                )
              ) : (
                <div className="chart-placeholder">
                  <p>Select a classification model and click APPLY to see results</p>
                </div>
              )}
            </div>

            {/* Graph Display Options - Inside Chart Panel */}
            {results && activeVisualization === 'spectrum' && (
              <div className="graph-display-options">
                <div className="display-checkboxes">
                  <label className="display-checkbox-label">
                    <input
                      type="checkbox"
                      checked={showPreprocessed}
                      onChange={(e) => setShowPreprocessed(e.target.checked)}
                    />
                    <span>Preprocessed Spectrum</span>
                  </label>
                  <label className="display-checkbox-label">
                    <input
                      type="checkbox"
                      checked={showDenoised}
                      onChange={(e) => setShowDenoised(e.target.checked)}
                    />
                    <span>Denoised Spectrum</span>
                  </label>
                  <label className="display-checkbox-label">
                    <input
                      type="checkbox"
                      checked={showClassification}
                      onChange={(e) => setShowClassification(e.target.checked)}
                    />
                    <span>Classification Reference</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Results and Controls */}
        <div className="results-control-panel">
          {/* Results Display */}
          <div className="results-panel">
            <h2>Classification Results</h2>
            {results?.warning && (
              <div className="result-warning">
                {results.warning}
              </div>
            )}
            <div className="result-item">
              <span className="result-label">Plastic Type:</span>
              <span className="result-value">
                {results ? results.plasticType : '-'}
              </span>
            </div>
            <div className="result-item">
              <span className="result-label">Correlation:</span>
              <span className="result-value">
                {results ? results.correlation.toFixed(4) : '-'}
              </span>
            </div>
          </div>

          {/* Configuration Info */}
          <div className="config-info">
            <h3>Current Configuration</h3>
            <div className="config-item">
              <span className="config-label">Membrane Filter:</span>
              <span className="config-value">{denoisingConfig.membraneFilter || 'Not Set'}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Denoising Model:</span>
              <span className="config-value">{denoisingConfig.denoisingModel || 'Not Set'}</span>
            </div>
            <div className="config-item">
              <span className="config-label">Classification:</span>
              <span className="config-value">{classificationModel || 'Not Set'}</span>
            </div>
          </div>

          {error && <p className="error-text">{error}</p>}

          {/* Action Buttons */}
          <div className="action-buttons">
            <button 
              className={`apply-button ${isApplied ? 'applied' : ''}`}
              onClick={handleApply}
              disabled={!classificationModel || isApplied || isProcessing}
            >
              {isProcessing ? (
                <>
                  <span className="loading-spinner"></span>
                  Classifying...
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

          <div className="completion-message">
            <p>🎉 Analysis Complete!</p>
            <p>You can review results or go back to any step to adjust parameters.</p>
          </div>
        </div>
      </div>

      {/* Clear Confirmation Modal */}
      {showClearModal && (
        <div className="modal-overlay" onClick={() => setShowClearModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Confirm Clear</h3>
            <p>Are you sure you want to clear the classification results?</p>
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

// Comparison Chart Component - Shows 3 spectrums
function ComparisonChart({
  wavenumbers,
  preprocessedIntensities,
  denoisedIntensities,
  classificationReference,
  camHeatmap = [],
  showHeatmap = true,
  showPreprocessed = true,
  showDenoised = true,
  showClassification = true
}) {

  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  const xMin = 650;
  const xMax = 4000;

  const allIntensities = [
    ...preprocessedIntensities,
    ...denoisedIntensities,
    ...(classificationReference || [])
  ];
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

  const classificationPath =
    classificationReference && classificationReference.length === wavenumbers.length
      ? wavenumbers
          .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(classificationReference[i])}`)
          .join(' ')
      : null;

  const hasHeatmap = Array.isArray(camHeatmap) && camHeatmap.length === wavenumbers.length;

  const heatColor = (value) => {
    const clamped = Math.max(0, Math.min(1, value || 0));
    const alpha = 0.25 + clamped * 0.6;
    const hue = 35 - clamped * 35;
    return `hsla(${hue}, 90%, 55%, ${alpha})`;
  };

  // Generate X-axis ticks
  const xTicks = [];
  const xTickCount = 5;
  for (let i = 0; i <= xTickCount; i++) {
    const value = xMin + (i / xTickCount) * (xMax - xMin);
    xTicks.push(value);
  }

  // Generate Y-axis ticks
  const yTicks = [];
  const yTickCount = 5;
  for (let i = 0; i <= yTickCount; i++) {
    const value = yMin + (i / yTickCount) * (yMax - yMin);
    yTicks.push(value);
  }

  return (
    <>
      <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg" style={{ width: '100%', height: 'auto' }}>
        <defs>
          <linearGradient id="preprocessedGradientPurpleStep4" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#7B2CBF" />
            <stop offset="100%" stopColor="#C77DFF" />
          </linearGradient>
          <linearGradient id="denoisedGradientGreen" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#059669" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
          <linearGradient id="classificationGradientYellow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>
        </defs>

        {/* Grid */}
        <g className="grid" opacity="0.1">
          {yTicks.map((yVal, i) => {
            const y = yScale(yVal);
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
          {xTicks.map((xVal, i) => {
            const x = xScale(xVal);
            return (
              <line
                key={`v-${i}`}
                x1={x}
                y1={padding.top}
                x2={x}
                y2={height - padding.bottom}
                stroke="#666"
                strokeWidth="1"
              />
            );
          })}
        </g>

        {/* Preprocessed Spectrum (purple) */}
        {showPreprocessed && (
          <path
            d={preprocessedPath}
            fill="none"
            stroke="url(#preprocessedGradientPurpleStep4)"
            strokeWidth="2"
            opacity="1"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Denoised Spectrum (green) */}
        {showDenoised && (
          <path
            d={denoisedPath}
            fill="none"
            stroke="url(#denoisedGradientGreen)"
            strokeWidth="2"
            opacity="1"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Classification Reference Spectrum (yellow) */}
        {showClassification && classificationPath && (
          <path
            d={classificationPath}
            fill="none"
            stroke="url(#classificationGradientYellow)"
            strokeWidth="2"
            opacity="1"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Heatmap overlay */}
        {hasHeatmap && showHeatmap &&
          wavenumbers.map((x, i) => (
            <circle
              key={`cam-${i}`}
              cx={xScale(x)}
              cy={yScale(denoisedIntensities[i])}
              r={3}
              fill={heatColor(camHeatmap[i])}
            />
          ))}

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

        {/* X-axis Ticks and Labels */}
        {xTicks.map((value, i) => {
          const x = xScale(value);
          return (
            <g key={`x-tick-${i}`}>
              <line
                x1={x}
                y1={height - padding.bottom}
                x2={x}
                y2={height - padding.bottom + 6}
                stroke="#666"
                strokeWidth="2"
              />
              <text
                x={x}
                y={height - padding.bottom + 20}
                textAnchor="middle"
                fill="#999"
                fontSize="11"
              >
                {value.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Y-axis Ticks and Labels */}
        {yTicks.map((value, i) => {
          const y = yScale(value);
          return (
            <g key={`y-tick-${i}`}>
              <line
                x1={padding.left - 6}
                y1={y}
                x2={padding.left}
                y2={y}
                stroke="#666"
                strokeWidth="2"
              />
              <text
                x={padding.left - 10}
                y={y + 4}
                textAnchor="end"
                fill="#999"
                fontSize="11"
              >
                {value.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(380, 20)">
          {showPreprocessed && (
            <>
              <circle cx="6" cy="0" r="6" fill="#7B2CBF"/>
              <text x="18" y="5" fill="#999" fontSize="12">Preprocessed</text>
            </>
          )}

          {showDenoised && (
            <>
              <circle cx="6" cy="15" r="6" fill="#059669"/>
              <text x="18" y="20" fill="#999" fontSize="12">Denoised</text>
            </>
          )}

          {showClassification && (
            <>
              <circle cx="6" cy="30" r="6" fill="#f59e0b"/>
              <text x="18" y="35" fill="#999" fontSize="12">Classification</text>
            </>
          )}

          {hasHeatmap && showHeatmap && (
            <>
              <circle cx="6" cy="45" r="6" fill={heatColor(1)} />
              <text x="18" y="50" fill="#facc15" fontSize="12">CAM intensity</text>
            </>
          )}
        </g>

        {/* Axis Labels */}
        <text x={width / 2} y={height - 10} textAnchor="middle" fill="#999" fontSize="14">
          Wavenumber (cm⁻¹)
        </text>
        <text x={20} y={height / 2} textAnchor="middle" fill="#999" fontSize="14"
          transform={`rotate(-90, 20, ${height / 2})`}>
          Intensity
        </text>
      </svg>
    </>
  );
}

export default Step4Classification;

function CamChart({ wavenumbers, inputIntensities, camHeatmap = [] }) {
  const width = 600;
  const height = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };

  if (!camHeatmap || camHeatmap.length !== wavenumbers.length) {
    return (
      <div className="chart-placeholder">
        <p>CAM heatmap unavailable. Run AlexNet or LeNet5 to generate.</p>
      </div>
    );
  }

  const xMin = 650;
  const xMax = 4000;
  const allIntensities = [...inputIntensities];
  const yMin = Math.min(...allIntensities);
  const yMax = Math.max(...allIntensities);

  const xScale = (x) => 
    padding.left + ((x - xMin) / (xMax - xMin)) * (width - padding.left - padding.right);
  
  const yScale = (y) => 
    height - padding.bottom - ((y - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);

  const heatColor = (value) => {
    const clamped = Math.max(0, Math.min(1, value || 0));
    const alpha = 0.3 + clamped * 0.6;
    const hue = 35 - clamped * 35;
    return `hsla(${hue}, 90%, 55%, ${alpha})`;
  };

  const inputPath = wavenumbers
    .map((x, i) => `${i === 0 ? 'M' : 'L'} ${xScale(x)} ${yScale(inputIntensities[i])}`)
    .join(' ');

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="spectrum-svg">
      {/* Input Spectrum */}
      <path 
        d={inputPath}
        fill="none"
        stroke="#0ea5e9"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Heatmap scatter */}
      {wavenumbers.map((x, i) => (
        <circle
          key={`cam-full-${i}`}
          cx={xScale(x)}
          cy={yScale(inputIntensities[i])}
          r={5}
          fill={heatColor(camHeatmap[i])}
        />
      ))}

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
      <g transform="translate(440, 30)">
        <line x1="0" y1="0" x2="30" y2="0" stroke="#0ea5e9" strokeWidth="2"/>
        <text x="35" y="5" fill="#999" fontSize="12">Input Spectrum</text>
        <circle cx="15" cy="20" r="8" fill={heatColor(1)} />
        <text x="35" y="25" fill="#facc15" fontSize="12">High CAM response</text>
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
