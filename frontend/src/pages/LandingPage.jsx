import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="landing-content">
        <div className="landing-text">
          <h1 className="landing-title">
            FTIR Microplastic<br />
            <span className="gradient-text">Analysis System</span>
          </h1>
          <p className="landing-description">
            Advanced deep learning denoising and classification for enhanced 
            microplastic FTIR identification
          </p>
          <button 
            className="start-button"
            onClick={() => navigate('/step1')}
          >
            START ANALYSIS
          </button>
          <div className="features">
            <div className="feature-item">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M9 11L12 14L22 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M21 12V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V5C3 3.89543 3.89543 3 5 3H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              <span>Senior Project | Group SL1</span>
            </div>
          </div>
        </div>
        <div className="landing-visual">
          <div className="spectrum-visualization">
            <svg viewBox="0 0 400 300" className="spectrum-svg">
              <defs>
                <linearGradient id="spectrumGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#7B2CBF" />
                  <stop offset="100%" stopColor="#C77DFF" />
                </linearGradient>
              </defs>
              
              <g className="spectrum-line">
                <path
                  d="M 20 250 Q 50 200, 80 220 T 140 180 T 200 160 T 260 140 T 320 120 T 380 100"
                  fill="none"
                  stroke="url(#spectrumGradient)"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
              </g>
              
              <g className="spectrum-dots">
                {[20, 80, 140, 200, 260, 320, 380].map((x, i) => {
                  const y = [250, 220, 180, 160, 140, 120, 100][i];
                  return (
                    <circle
                      key={i}
                      cx={x}
                      cy={y}
                      r="4"
                      fill="#C77DFF"
                      className="spectrum-dot"
                      style={{ animationDelay: `${i * 0.2}s` }}
                    />
                  );
                })}
              </g>
              
              <g className="axis-lines" opacity="0.3">
                <line x1="20" y1="20" x2="20" y2="280" stroke="#666" strokeWidth="1" />
                <line x1="20" y1="280" x2="380" y2="280" stroke="#666" strokeWidth="1" />
              </g>
            </svg>
          </div>
        </div>
      </div>
      <div className="landing-background">
        <div className="orb orb-1"></div>
        <div className="orb orb-2"></div>
        <div className="orb orb-3"></div>
      </div>
    </div>
  );
}

export default LandingPage;
