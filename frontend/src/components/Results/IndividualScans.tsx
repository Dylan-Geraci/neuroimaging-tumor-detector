import { useState } from 'react';
import type { IndividualPrediction } from '../../api/types';
import { formatClassName } from '../../utils';

interface Props {
  predictions: IndividualPrediction[];
}

export default function IndividualScans({ predictions }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="individual-scans">
      <button className={`expand-btn${expanded ? ' expanded' : ''}`} onClick={() => setExpanded(!expanded)}>
        <span>{expanded ? 'Hide Individual Scans' : 'View Individual Scans'}</span>
        <svg
          className="expand-icon"
          width="16"
          height="16"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="scans-container" style={{ marginTop: '1rem' }}>
          {predictions.map((pred, index) => {
            if (pred.error) {
              return (
                <div key={pred.filename} className="scan-card error">
                  <div className="scan-details">
                    <div className="scan-filename">{pred.filename}</div>
                    <div className="scan-error">Error: {pred.error}</div>
                  </div>
                </div>
              );
            }

            return (
              <div
                key={pred.filename}
                className="scan-card"
                style={{ animationDelay: `${index * 0.06}s` }}
              >
                <div className="scan-thumbnail">
                  <img src={pred.images.original} alt={pred.filename} />
                </div>
                <div className="scan-details">
                  <div className="scan-filename">{pred.filename}</div>
                  <div className="scan-prediction">{formatClassName(pred.class)}</div>
                  <div className="scan-confidence">{(pred.confidence * 100).toFixed(1)}%</div>
                </div>
                <div className="scan-overlay">
                  <img src={pred.images.overlay} alt="Overlay" />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
