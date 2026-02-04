export interface PredictionResult {
  class: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface PredictionImages {
  original: string;
  heatmap: string;
  overlay: string;
}

export interface SinglePredictionResponse {
  success: boolean;
  prediction: PredictionResult;
  images: PredictionImages;
}

export interface IndividualPrediction {
  filename: string;
  class: string;
  class_index: number;
  confidence: number;
  probabilities: Record<string, number>;
  images: PredictionImages;
  error?: string;
}

export interface AggregatedPrediction {
  class: string;
  confidence: number;
  probabilities: Record<string, number>;
  agreement_score: number;
}

export interface BatchPredictionResponse {
  success: boolean;
  batch_size: number;
  processed_count: number;
  individual_predictions: IndividualPrediction[];
  aggregated_prediction: AggregatedPrediction;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  gradcam_loaded: boolean;
  device: string | null;
  classes: string[];
}

export type PredictionResponse = SinglePredictionResponse | BatchPredictionResponse;

export function isBatchResponse(r: PredictionResponse): r is BatchPredictionResponse {
  return 'aggregated_prediction' in r;
}
