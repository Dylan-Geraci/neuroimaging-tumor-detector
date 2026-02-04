import { useEffect, useRef, useState } from 'react';
import Header from './components/Layout/Header';
import Footer from './components/Layout/Footer';
import UploadArea from './components/Upload/UploadArea';
import FilePreview from './components/Upload/FilePreview';
import UploadProgress from './components/Upload/UploadProgress';
import ResultsPanel from './components/Results/ResultsPanel';
import HistoryPanel from './components/History/HistoryPanel';
import { useFileSelection } from './hooks/useFileSelection';
import { usePrediction } from './hooks/usePrediction';
import { checkHealth } from './api/client';

export default function App() {
  const { files, addFiles, removeFile, clearFiles } = useFileSelection();
  const { phase, progress, result, error, run, reset } = usePrediction();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [healthWarning, setHealthWarning] = useState<string | null>(null);
  const [toasts, setToasts] = useState<{ id: number; message: string; type: string }[]>([]);

  const showToast = (message: string, type = 'error', duration = 5000) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    if (duration > 0) {
      setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), duration);
    }
  };

  useEffect(() => {
    checkHealth().catch(() => {
      setHealthWarning('Could not connect to the prediction API. Please ensure the backend is running.');
    });
  }, []);

  useEffect(() => {
    if (error) {
      showToast(error);
      handleReset();
    }
  }, [error]);

  const handleUpload = () => {
    if (files.length === 0) return;
    run(files);
  };

  const handleReset = () => {
    reset();
    clearFiles();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleAddMore = () => fileInputRef.current?.click();

  const handleHiddenInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addFiles(e.target.files);
    e.target.value = '';
  };

  const showUpload = phase === 'idle';
  const showLoading = phase === 'uploading' || phase === 'analyzing';
  const showResults = phase === 'done' && result;

  return (
    <>
      {/* Toast container */}
      <div className="toast-container">
        {toasts.map((t) => (
          <div key={t.id} className={`toast${t.type === 'warning' ? ' toast-warning' : t.type === 'success' ? ' toast-success' : ''}`}>
            <span className="toast-message">{t.message}</span>
            <button className="toast-close" onClick={() => setToasts((prev) => prev.filter((x) => x.id !== t.id))}>
              &times;
            </button>
          </div>
        ))}
      </div>

      <div className="container">
        {/* Health Banner */}
        {healthWarning && (
          <div className="health-banner">
            <span className="health-banner-message">{healthWarning}</span>
            <button className="health-banner-close" onClick={() => setHealthWarning(null)}>
              &times;
            </button>
          </div>
        )}

        <Header />

        {/* Medical Disclaimer */}
        <div className="disclaimer-banner">
          <div className="disclaimer-icon">⚠️</div>
          <div className="disclaimer-content">
            <strong>Educational Use Only:</strong> This system is for research and
            educational purposes. Not intended for clinical diagnosis. Always consult
            qualified medical professionals for health decisions.
          </div>
        </div>

        <main>
          {/* Hidden extra file input for "Add More" */}
          <input ref={fileInputRef} type="file" accept="image/*" multiple hidden onChange={handleHiddenInput} />

          {showUpload && (
            <section className="upload-section">
              <UploadArea onFiles={addFiles} />
              <FilePreview
                files={files}
                onRemove={removeFile}
                onClear={clearFiles}
                onUpload={handleUpload}
                onAddMore={handleAddMore}
              />
            </section>
          )}

          {showLoading && (
            <UploadProgress
              phase={phase as 'uploading' | 'analyzing'}
              progress={progress}
              fileCount={files.length}
            />
          )}

          {showResults && <ResultsPanel result={result} onReset={handleReset} />}

          {showUpload && <HistoryPanel />}
        </main>

        <Footer />
      </div>
    </>
  );
}
