import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Step4Classification.css';

const CLASSIFICATION_MODELS = [
  { value: 'disable', label: <>Correlation <br/> (Library Search)</> },
  { value: 'LeNet5', label: 'LeNet5' },
  { value: 'AlexNet', label: 'AlexNet' }
];

function Step4Classification({
  uploadedFile,
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
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [results, setResults] = useState(null);
  const [activeVisualization, setActiveVisualization] = useState('spectrum');
  const [showPreprocessed, setShowPreprocessed] = useState(true);
  const [showDenoised, setShowDenoised] = useState(true);
  const [showClassification, setShowClassification] = useState(true);

  // Restore results from spectralData when component mounts or when classification data changes
  useEffect(() => {
    if (spectralData.classificationResult && isApplied) {
      const data = spectralData.classificationResult;
      setResults({
        plasticType: data.plastic_type,
        correlation: data.correlation,
        cleanSpectrum: data.clean_spectrum,
        classificationReference: data.reference_spectrum || [],
        warning: data.warning || null,
        camHeatmap: data.cam_heatmap || []
      });
    } else if (!isApplied) {
      setResults(null);
    }
  }, [spectralData.classificationResult, isApplied]);

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

  const generateReportCanvas = async () => {
    // Dynamically import required library
    const html2canvas = (await import('html2canvas')).default;

    // Create a temporary div for PDF generation
    const pdfContainer = document.createElement('div');
    pdfContainer.style.width = '720px'; // Reduced to fit A4 better
    pdfContainer.style.padding = '12px';
    pdfContainer.style.fontFamily = 'Arial, sans-serif';
    pdfContainer.style.backgroundColor = 'white';

    // Header
    const header = document.createElement('div');
    header.style.borderBottom = '3px solid #7B2CBF';
    header.style.paddingBottom = '8px';
    header.style.marginBottom = '8px';
    header.style.display = 'flex';
    header.style.alignItems = 'center';
    header.style.gap = '15px';

    // Left side - Logo (15%)
    const logoContainer = document.createElement('div');
    logoContainer.style.width = '15%';
    logoContainer.style.display = 'flex';
    logoContainer.style.justifyContent = 'center';
    logoContainer.style.alignItems = 'center';

    const logo = document.createElement('img');
    logo.src = '/siit_logo.png';
    logo.alt = 'SIIT Logo';
    logo.style.width = '65px';
    logo.style.height = 'auto';
    logo.onerror = () => {
      logo.style.display = 'none';
      const logoText = document.createElement('div');
      logoText.textContent = 'SIIT';
      logoText.style.fontSize = '28px';
      logoText.style.fontWeight = 'bold';
      logoText.style.color = '#7B2CBF';
      logoContainer.appendChild(logoText);
    };
    logoContainer.appendChild(logo);

    // Right side - Info (85%)
    const infoContainer = document.createElement('div');
    infoContainer.style.width = '85%';
    infoContainer.style.textAlign = 'left';

    const title = document.createElement('h5');
    title.style.color = '#2c3e50';
    title.style.fontSize = '0.95em';
    title.style.fontWeight = 'bold';
    title.style.margin = '0 0 4px 0';
    title.textContent = 'Deep Learning Denoising for Enhanced Microplastic FTIR Identification';
    infoContainer.appendChild(title);

    const fileName = document.createElement('p');
    fileName.style.margin = '2px 0';
    fileName.style.fontSize = '0.65em';
    fileName.style.color = '#2c3e50';
    fileName.innerHTML = `<strong>Source File:</strong> ${uploadedFile?.name || 'Uploaded Spectrum'}`;
    infoContainer.appendChild(fileName);

    const today = new Date().toLocaleDateString('en-GB', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
    const dateIssued = document.createElement('p');
    dateIssued.style.margin = '2px 0';
    dateIssued.style.fontSize = '0.65em';
    dateIssued.style.color = '#2c3e50';
    dateIssued.innerHTML = `<strong>Analysis Date:</strong> ${today}`;
    infoContainer.appendChild(dateIssued);

    const membraneInfo = document.createElement('p');
    membraneInfo.style.margin = '2px 0';
    membraneInfo.style.fontSize = '0.65em';
    membraneInfo.style.color = '#2c3e50';
    membraneInfo.innerHTML = `<strong>Membrane Filter Type:</strong> ${denoisingConfig.membraneFilter || 'Not specified'}`;
    infoContainer.appendChild(membraneInfo);

    header.appendChild(logoContainer);
    header.appendChild(infoContainer);
    pdfContainer.appendChild(header);

    // Content sections
    const content = document.createElement('div');
    content.style.display = 'flex';
    content.style.flexDirection = 'column';
    content.style.alignItems = 'center';

    const sectionStyle = {
      marginBottom: '2px',
      paddingBottom: '2px',
      paddingTop: '1px',
      borderBottom: '2px solid #ddd',
      width: '100%',
    };

    const noBorderStyle = {
      marginBottom: '2px',
      paddingTop: '1px',
      width: '100%',
    };

    const headingStyle = {
      color: '#2c3e50',
      marginBottom: '2px',
      textAlign: 'left',
      fontSize: '0.85em',
      fontWeight: 'bold',
      borderBottom: '2px solid #ddd',
      paddingBottom: '1px',
    };

    const canvasWidth = 670;
    const canvasHeight = 180;
    const dpi = 2; // Higher DPI for sharper rendering

    // Helper function to create chart canvas
    const createChartSection = (title, wavenumbers, intensities, color, showBorder = true) => {
        if (!intensities || intensities.length === 0) return null;

        const section = document.createElement('div');
        Object.assign(section.style, showBorder ? sectionStyle : noBorderStyle);

        const heading = document.createElement('h4');
        Object.assign(heading.style, headingStyle);
        heading.textContent = title;
        section.appendChild(heading);

        const canvas = document.createElement('canvas');
        canvas.width = canvasWidth * dpi;
        canvas.height = canvasHeight * dpi;
        canvas.style.width = `${canvasWidth}px`;
        canvas.style.height = `${canvasHeight}px`;
        canvas.style.paddingTop = '1px';
        canvas.style.paddingBottom = '1px';

        // Draw chart using canvas 2D context
        const ctx = canvas.getContext('2d');
        ctx.scale(dpi, dpi);
        const padding = { top: 25, right: 30, bottom: 35, left: 50 };
        const xMin = 650;
        const xMax = 4000;
        const yMin = Math.min(...intensities);
        const yMax = Math.max(...intensities);

        const xScale = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (canvasWidth - padding.left - padding.right);
        const yScale = (y) => canvasHeight - padding.bottom - ((y - yMin) / (yMax - yMin)) * (canvasHeight - padding.top - padding.bottom);

        // Draw grid
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
          const y = yScale(yMin + (i / 5) * (yMax - yMin));
          ctx.beginPath();
          ctx.moveTo(padding.left, y);
          ctx.lineTo(canvasWidth - padding.right, y);
          ctx.stroke();

          const x = xScale(xMin + (i / 5) * (xMax - xMin));
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, canvasHeight - padding.bottom);
          ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, canvasHeight - padding.bottom);
        ctx.lineTo(canvasWidth - padding.right, canvasHeight - padding.bottom);
        ctx.stroke();

        // Draw spectrum line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        wavenumbers.forEach((x, i) => {
          const px = xScale(x);
          const py = yScale(intensities[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Draw axis labels
        ctx.fillStyle = '#666';
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
          const value = xMin + (i / 5) * (xMax - xMin);
          const x = xScale(value);
          ctx.fillText(value.toFixed(0), x, canvasHeight - padding.bottom + 10);
        }

        ctx.textAlign = 'end';
        ctx.font = '8px Arial';
        for (let i = 0; i <= 5; i++) {
          const value = yMin + (i / 5) * (yMax - yMin);
          const y = yScale(value);
          ctx.fillText(value.toFixed(2), padding.left - 5, y + 3);
        }

        // Axis titles
        ctx.font = '9px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Wavenumber (cm⁻¹)', canvasWidth / 2, canvasHeight - 3);

        ctx.save();
        ctx.translate(12, canvasHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Intensity', 0, 0);
        ctx.restore();

        section.appendChild(canvas);
        return section;
      };

      // Add Input Spectrum (using purple color from UI)
      const inputSection = createChartSection(
        'Input Spectrum',
        spectralData.wavenumbers,
        spectralData.originalIntensities,
        '#7B2CBF' // Purple from UI
      );
      if (inputSection) content.appendChild(inputSection);

      // Add Preprocessed Spectrum (using purple gradient from UI)
      const preprocessedSection = createChartSection(
        'Preprocessed Spectrum',
        spectralData.wavenumbers,
        spectralData.preprocessedIntensities,
        '#9333ea' // Mid-purple between #7B2CBF and #C77DFF
      );
      if (preprocessedSection) content.appendChild(preprocessedSection);

      // Add Denoised Spectrum (using green from UI)
      const denoisedSection = createChartSection(
        'Denoised Spectrum',
        spectralData.wavenumbers,
        spectralData.denoisedIntensities,
        '#059669' // Green from UI
      );
      if (denoisedSection) content.appendChild(denoisedSection);

      // Add Classification Comparison with legend
      if (spectralData.denoisedIntensities && spectralData.preprocessedIntensities && results.classificationReference) {
        const classSection = document.createElement('div');
        Object.assign(classSection.style, noBorderStyle);

        const heading = document.createElement('h4');
        Object.assign(heading.style, headingStyle);
        heading.textContent = 'Classification Compared';
        classSection.appendChild(heading);

        const canvas = document.createElement('canvas');
        canvas.width = canvasWidth * dpi;
        canvas.height = canvasHeight * dpi;
        canvas.style.width = `${canvasWidth}px`;
        canvas.style.height = `${canvasHeight}px`;
        canvas.style.paddingTop = '1px';
        canvas.style.paddingBottom = '1px';

        const ctx = canvas.getContext('2d');
        ctx.scale(dpi, dpi);
        const padding = { top: 25, right: 30, bottom: 35, left: 50 };
        const xMin = 650;
        const xMax = 4000;
        const allIntensities = [...spectralData.preprocessedIntensities, ...spectralData.denoisedIntensities, ...results.classificationReference];
        const yMin = Math.min(...allIntensities);
        const yMax = Math.max(...allIntensities);

        const xScale = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * (canvasWidth - padding.left - padding.right);
        const yScale = (y) => canvasHeight - padding.bottom - ((y - yMin) / (yMax - yMin)) * (canvasHeight - padding.top - padding.bottom);

        // Draw grid
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {
          const y = yScale(yMin + (i / 5) * (yMax - yMin));
          ctx.beginPath();
          ctx.moveTo(padding.left, y);
          ctx.lineTo(canvasWidth - padding.right, y);
          ctx.stroke();

          const x = xScale(xMin + (i / 5) * (xMax - xMin));
          ctx.beginPath();
          ctx.moveTo(x, padding.top);
          ctx.lineTo(x, canvasHeight - padding.bottom);
          ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, canvasHeight - padding.bottom);
        ctx.lineTo(canvasWidth - padding.right, canvasHeight - padding.bottom);
        ctx.stroke();

        // Draw Preprocessed spectrum (purple from UI)
        ctx.strokeStyle = '#9333ea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        spectralData.wavenumbers.forEach((x, i) => {
          const px = xScale(x);
          const py = yScale(spectralData.preprocessedIntensities[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Draw Denoised spectrum (green from UI)
        ctx.strokeStyle = '#059669';
        ctx.lineWidth = 2;
        ctx.beginPath();
        spectralData.wavenumbers.forEach((x, i) => {
          const px = xScale(x);
          const py = yScale(spectralData.denoisedIntensities[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Draw Reference spectrum (orange from UI)
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        spectralData.wavenumbers.forEach((x, i) => {
          const px = xScale(x);
          const py = yScale(results.classificationReference[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // Draw axis labels
        ctx.fillStyle = '#666';
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        for (let i = 0; i <= 5; i++) {
          const value = xMin + (i / 5) * (xMax - xMin);
          const x = xScale(value);
          ctx.fillText(value.toFixed(0), x, canvasHeight - padding.bottom + 10);
        }

        ctx.textAlign = 'end';
        ctx.font = '8px Arial';
        for (let i = 0; i <= 5; i++) {
          const value = yMin + (i / 5) * (yMax - yMin);
          const y = yScale(value);
          ctx.fillText(value.toFixed(2), padding.left - 5, y + 3);
        }

        // Axis titles
        ctx.font = '9px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Wavenumber (cm⁻¹)', canvasWidth / 2, canvasHeight - 3);

        ctx.save();
        ctx.translate(12, canvasHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Intensity', 0, 0);
        ctx.restore();

        // Draw legend - compact version
        const legendX = canvasWidth - 140;
        const legendY = 35;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.fillRect(legendX, legendY, 130, 55);
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.strokeRect(legendX, legendY, 130, 55);

        // Preprocessed legend (purple)
        ctx.strokeStyle = '#9333ea';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX + 6, legendY + 10);
        ctx.lineTo(legendX + 20, legendY + 10);
        ctx.stroke();
        ctx.fillStyle = '#333';
        ctx.font = '7.5px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Preprocessed', legendX + 24, legendY + 12);

        // Denoised legend (green)
        ctx.strokeStyle = '#059669';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX + 6, legendY + 26);
        ctx.lineTo(legendX + 20, legendY + 26);
        ctx.stroke();
        ctx.fillStyle = '#333';
        ctx.fillText('Denoised', legendX + 24, legendY + 28);

        // Reference legend (orange)
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(legendX + 6, legendY + 42);
        ctx.lineTo(legendX + 20, legendY + 42);
        ctx.stroke();
        ctx.fillStyle = '#333';
        ctx.fillText(`${results.plasticType || 'Plastic'} Reference`, legendX + 24, legendY + 44);

        classSection.appendChild(canvas);

        // Add classification results
        const resultDetails = document.createElement('div');
        resultDetails.style.fontSize = '0.75em';
        resultDetails.style.textAlign = 'left';
        resultDetails.style.marginTop = '3px';
        resultDetails.style.color = '#2c3e50';
        resultDetails.innerHTML = `
          <p style="margin: 2px 0; color: #2c3e50;"><strong>Predicted Plastic Type:</strong> <span style="color: #7B2CBF; font-weight: bold;">${results.plasticType || 'N/A'}</span></p>
          <p style="margin: 2px 0; color: #2c3e50;"><strong>Correlation:</strong> ${results.correlation?.toFixed(4) || 'N/A'}</p>
        `;
        classSection.appendChild(resultDetails);

        content.appendChild(classSection);
      }

      pdfContainer.appendChild(content);

      // Footer
      const footer = document.createElement('div');
      footer.style.borderTop = '1px solid #ccc';
      footer.style.paddingTop = '3px';
      footer.style.marginTop = '4px';
      footer.style.fontSize = '0.65em';
      footer.style.color = '#7f8c8d';
      footer.style.textAlign = 'center';
      footer.innerHTML = 'Generated By SL1 | Computer Engineering Senior Project';
      pdfContainer.appendChild(footer);

    // Append to body
    document.body.appendChild(pdfContainer);

    // Generate canvas with higher quality
    const pdfCanvas = await html2canvas(pdfContainer, {
      scale: 4,
      useCORS: true,
      allowTaint: true,
      scrollY: 0,
      logging: false,
      backgroundColor: '#ffffff',
    });

    // Remove temporary container
    document.body.removeChild(pdfContainer);

    return pdfCanvas;
  };

  const handlePrintReport = async () => {
    setIsProcessing(true);
    try {
      const canvas = await generateReportCanvas();
      const imgData = canvas.toDataURL('image/png');

      // Show preview modal
      setPreviewImage(imgData);
      setShowPreviewModal(true);
    } catch (error) {
      console.error('Error generating report preview:', error);
      alert('Failed to generate report preview. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadPDF = async () => {
    try {
      const jsPDF = (await import('jspdf')).default;
      const canvas = await generateReportCanvas();
      const imgData = canvas.toDataURL('image/png');

      const pdf = new jsPDF('p', 'pt', 'a4');
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save('FTIR_Analysis_Report.pdf');

      setShowPreviewModal(false);
    } catch (error) {
      console.error('Error downloading PDF:', error);
      alert('Failed to download PDF. Please try again.');
    }
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
                    plasticType={results.plasticType}
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
                    <span>{results?.plasticType ? `${results.plasticType} Reference Spectrum` : 'Classification Reference'}</span>
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
            {isApplied && results && (
              <button
                className="print-report-button"
                onClick={handlePrintReport}
              >
                📄 Print Report
              </button>
            )}
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

      {/* Report Preview Modal */}
      {showPreviewModal && (
        <div className="modal-overlay" onClick={() => setShowPreviewModal(false)}>
          <div className="modal-content preview-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Report Preview</h3>
            <div className="preview-container">
              {previewImage && (
                <img src={previewImage} alt="Report Preview" style={{ width: '100%', border: '1px solid #ddd' }} />
              )}
            </div>
            <div className="modal-buttons">
              <button className="modal-button confirm" onClick={handleDownloadPDF}>
                Download PDF
              </button>
              <button className="modal-button cancel" onClick={() => setShowPreviewModal(false)}>
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
  showClassification = true,
  plasticType = null
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
            opacity="0.8"
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
            opacity="0.8"
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
            opacity="0.8"
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
              <text x="18" y="35" fill="#999" fontSize="12">{plasticType ? `${plasticType} Reference` : 'Classification'}</text>
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
