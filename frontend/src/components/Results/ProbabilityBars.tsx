import { useEffect, useState } from 'react';
import { formatClassName } from '../../utils';

interface Props {
  probabilities: Record<string, number>;
  title?: string;
}

export default function ProbabilityBars({ probabilities, title = 'Class Probabilities' }: Props) {
  const [animated, setAnimated] = useState(false);
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 50);
    return () => clearTimeout(t);
  }, [probabilities]);

  return (
    <div className="probabilities">
      <h3>{title}</h3>
      <div className="probability-bars">
        {sorted.map(([cls, prob], i) => (
          <div key={cls} className="prob-item">
            <div className="prob-label">
              <span>{formatClassName(cls)}</span>
              <span className="prob-value">{(prob * 100).toFixed(2)}%</span>
            </div>
            <div className="prob-bar-container">
              <div
                className="prob-bar"
                style={{
                  width: animated ? `${prob * 100}%` : '0%',
                  transitionDelay: `${i * 80}ms`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
