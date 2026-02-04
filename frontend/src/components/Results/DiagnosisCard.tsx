import { formatClassName } from '../../utils';

interface Props {
  label: string;
  className: string;
  confidence: number;
  batchInfo?: {
    scanCount: number;
    agreementScore: number;
  };
}

function getAgreementClass(score: number): string {
  if (score === 1) return 'full-agreement';
  if (score >= 0.75) return 'high-agreement';
  if (score >= 0.5) return 'medium-agreement';
  return 'low-agreement';
}

export default function DiagnosisCard({ label, className, confidence, batchInfo }: Props) {
  return (
    <div className="prediction-box">
      <h2>{label}</h2>
      <div className="prediction-result">
        <div className="prediction-class">{formatClassName(className)}</div>
        <div className="confidence">{(confidence * 100).toFixed(2)}% Confidence</div>
      </div>
      {batchInfo && (
        <div className="batch-info">
          <span className="scan-count">{batchInfo.scanCount} scans analyzed</span>
          <span className={`agreement-indicator ${getAgreementClass(batchInfo.agreementScore)}`}>
            {(batchInfo.agreementScore * 100).toFixed(0)}% Agreement
          </span>
        </div>
      )}
    </div>
  );
}
