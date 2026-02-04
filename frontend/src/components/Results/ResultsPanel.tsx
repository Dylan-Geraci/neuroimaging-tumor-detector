import type { PredictionResponse } from '../../api/types';
import { isBatchResponse } from '../../api/types';
import DiagnosisCard from './DiagnosisCard';
import ProbabilityBars from './ProbabilityBars';
import GradCamViewer from './GradCamViewer';
import IndividualScans from './IndividualScans';
import ExportButtons from '../Export/ExportButtons';

interface Props {
  result: PredictionResponse;
  onReset: () => void;
}

export default function ResultsPanel({ result, onReset }: Props) {
  if (isBatchResponse(result)) {
    const { aggregated_prediction, individual_predictions, processed_count } = result;
    return (
      <section className="results-section">
        <DiagnosisCard
          label="Aggregated Diagnosis"
          className={aggregated_prediction.class}
          confidence={aggregated_prediction.confidence}
          batchInfo={{
            scanCount: processed_count,
            agreementScore: aggregated_prediction.agreement_score,
          }}
        />
        <ProbabilityBars
          probabilities={aggregated_prediction.probabilities}
          title="Aggregated Class Probabilities"
        />
        <ExportButtons result={result} />
        <IndividualScans predictions={individual_predictions} />
        <button className="reset-btn" onClick={onReset}>
          Start Over
        </button>
        <p className="reset-hint">Upload new scans for classification</p>
      </section>
    );
  }

  // Single prediction
  const { prediction, images } = result;
  return (
    <section className="results-section">
      <DiagnosisCard label="Diagnosis" className={prediction.class} confidence={prediction.confidence} />
      <ProbabilityBars probabilities={prediction.probabilities} />
      <ExportButtons result={result} />
      <GradCamViewer images={images} />
      <button className="reset-btn" onClick={onReset}>
        Start Over
      </button>
      <p className="reset-hint">Upload new scans for classification</p>
    </section>
  );
}
