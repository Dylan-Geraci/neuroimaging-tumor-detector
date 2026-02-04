import { useState, useCallback } from 'react';
import { predictSingle, predictBatch } from '../api/client';
import type { PredictionResponse } from '../api/types';

type Phase = 'idle' | 'uploading' | 'analyzing' | 'done' | 'error';

export function usePrediction() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (files: File[]) => {
    setPhase('uploading');
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const onProgress = (pct: number) => setProgress(Math.round(pct));

      let data: PredictionResponse;
      if (files.length === 1) {
        data = await predictSingle(files[0], onProgress);
      } else {
        data = await predictBatch(files, onProgress);
      }

      setPhase('analyzing');
      // Brief delay so the user sees the analyzing phase
      await new Promise((r) => setTimeout(r, 400));

      if (data.success) {
        setResult(data);
        setPhase('done');
      } else {
        throw new Error('Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setPhase('error');
    }
  }, []);

  const reset = useCallback(() => {
    setPhase('idle');
    setProgress(0);
    setResult(null);
    setError(null);
  }, []);

  return { phase, progress, result, error, run, reset };
}
