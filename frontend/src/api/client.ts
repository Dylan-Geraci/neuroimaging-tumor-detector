import type { SinglePredictionResponse, BatchPredictionResponse, HealthResponse } from './types';

const API_URL = '';

export async function checkHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_URL}/health`);
  if (!resp.ok) throw new Error('Health check failed');
  return resp.json();
}

export function xhrUpload<T>(
  url: string,
  formData: FormData,
  onProgress: (pct: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        onProgress((e.loaded / e.total) * 100);
      }
    });

    xhr.addEventListener('load', () => {
      let data: unknown;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error('Invalid response from server'));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(data as T);
      } else {
        reject(new Error((data as { detail?: string }).detail || `Request failed (${xhr.status})`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Network error — please check your connection.'));
    });

    xhr.send(formData);
  });
}

export function predictSingle(
  file: File,
  onProgress: (pct: number) => void,
): Promise<SinglePredictionResponse> {
  const fd = new FormData();
  fd.append('file', file);
  return xhrUpload(`${API_URL}/predict`, fd, onProgress);
}

export function predictBatch(
  files: File[],
  onProgress: (pct: number) => void,
): Promise<BatchPredictionResponse> {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  return xhrUpload(`${API_URL}/predict/batch`, fd, onProgress);
}
