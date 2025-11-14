import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Sidebar.css';

function Sidebar({ currentStep, progressPercentage, onStepClick, isMobile, appliedSteps }) {
  const navigate = useNavigate();
  const location = useLocation();

  const steps = [
    { id: 0, name: 'Input Spectrum', path: '/step1', key: 'step1' },
    { id: 1, name: 'Preprocessed Spectrum', path: '/step2', key: 'step2' },
    { id: 2, name: 'Denoising Spectrum', path: '/step3', key: 'step3' },
    { id: 3, name: 'Classification', path: '/step4', key: 'step4' }
  ];

  const handleStepClick = (step) => {
    // Can navigate to any step that has been applied or current step
    const stepIndex = steps.findIndex(s => s.id === step.id);
    const canNavigate = stepIndex === 0 || appliedSteps[steps[stepIndex - 1].key];
    
    if (canNavigate || step.path === location.pathname) {
      navigate(step.path);
      onStepClick(step.id);
    }
  };

  const isStepAccessible = (step) => {
    if (step.id === 0) return true;
    const stepIndex = steps.findIndex(s => s.id === step.id);
    return appliedSteps[steps[stepIndex - 1].key];
  };

  const isStepCompleted = (step) => {
    return appliedSteps[step.key];
  };

  const isStepActive = (step) => {
    return location.pathname === step.path;
  };

  return (
    <aside className={`sidebar ${isMobile ? 'mobile' : ''}`}>
      <div className="sidebar-header">
        <h2>Progress</h2>
        <div className="progress-info">
          <span className="progress-text">{Math.round(progressPercentage)}% Complete</span>
        </div>
      </div>

      <div className="progress-bar-container">
        <div 
          className="progress-bar-fill" 
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      <nav className="steps-list">
        {steps.map((step) => {
          const accessible = isStepAccessible(step);
          const completed = isStepCompleted(step);
          const active = isStepActive(step);

          return (
            <button
              key={step.id}
              className={`step-item ${active ? 'active' : ''} ${completed ? 'completed' : ''} ${!accessible ? 'disabled' : ''}`}
              onClick={() => handleStepClick(step)}
              disabled={!accessible && !active}
            >
              <div className="step-number">
                {completed ? (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path 
                      d="M7 10L9 12L13 8" 
                      stroke="currentColor" 
                      strokeWidth="2" 
                      strokeLinecap="round" 
                      strokeLinejoin="round"
                    />
                    <circle 
                      cx="10" 
                      cy="10" 
                      r="9" 
                      stroke="currentColor" 
                      strokeWidth="2"
                    />
                  </svg>
                ) : (
                  <span>{step.id + 1}</span>
                )}
              </div>
              <div className="step-info">
                <span className="step-name">{step.name}</span>
                {active && <span className="step-status">Current</span>}
                {completed && !active && <span className="step-status completed">Applied</span>}
              </div>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="footer-info">
          <p>Deep Learning Denoising</p>
          <p>© 2025 SIIT</p>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
