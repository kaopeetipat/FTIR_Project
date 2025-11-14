import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './Header.css';

function Header({ isMobile, currentStep, onStepChange, appliedSteps }) {
  const navigate = useNavigate();
  const location = useLocation();

  const stepPaths = ['/', '/step1', '/step2', '/step3', '/step4'];

  const handleBack = () => {
    const currentPath = location.pathname;
    const currentIndex = stepPaths.indexOf(currentPath);
    
    if (currentIndex > 0) {
      const previousPath = stepPaths[currentIndex - 1];
      navigate(previousPath);
      if (currentIndex > 1) {
        onStepChange(currentIndex - 2);
      }
    }
  };

  const isBackDisabled = location.pathname === '/' || location.pathname === '/step1';

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-container">
          <div className="logo-gradient">SIIT</div>
        </div>
        <h1 className="header-title">
          FTIR Microplastic Analysis
        </h1>
      </div>

      <div className="header-right">
        <button 
          className="back-button"
          onClick={handleBack}
          disabled={isBackDisabled}
          title={isBackDisabled ? "Already at start" : "Go back to previous step"}
        >
          <svg 
            width="20" 
            height="20" 
            viewBox="0 0 20 20" 
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              d="M12.5 15L7.5 10L12.5 5" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
          </svg>
          <span>Back</span>
        </button>
      </div>
    </header>
  );
}

export default Header;
