import type { SinglePredictionResponse, BatchPredictionResponse, HealthResponse } from './types';

// Use environment variable for API URL, fallback to empty string for dev (proxy)
const API_URL = import.meta.env.VITE_API_URL || '';
const API_KEY = import.meta.env.VITE_API_KEY || '';

function getHeaders(): HeadersInit {
  const headers: HeadersInit = {};
  if (API_KEY) {
    headers['X-API-Key'] = API_KEY;
  }
  return headers;
}

export async function checkHealth(): Promise<HealthResponse> {
  const resp = await fetch(`${API_URL}/health`, {
    headers: getHeaders(),
  });
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

    // Add API key header if configured
    if (API_KEY) {
      xhr.setRequestHeader('X-API-Key', API_KEY);
    }

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
      } else if (xhr.status === 503 || xhr.status === 502 || xhr.status === 0) {
        reject(new Error('Backend unavailable. See README for local setup.'));
      } else {
        reject(new Error((data as { detail?: string }).detail || `Request failed (${xhr.status})`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Backend unavailable. See README for local setup.'));
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
