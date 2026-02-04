interface Props {
  phase: 'uploading' | 'analyzing';
  progress: number;
  fileCount: number;
}

export default function UploadProgress({ phase, progress, fileCount }: Props) {
  if (phase === 'uploading') {
    const label = fileCount === 1 ? 'Uploading scan...' : `Uploading ${fileCount} scans...`;
    return (
      <div className="loading">
        <p className="loading-text">{label}</p>
        <div className="progress-bar-track">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
        </div>
        <p className="progress-percent">{progress}%</p>
      </div>
    );
  }

  const label = fileCount === 1 ? 'Analyzing scan...' : `Analyzing ${fileCount} scans...`;
  return (
    <div className="loading">
      <div className="spinner">
        <span></span>
        <span></span>
        <span></span>
      </div>
      <p className="loading-text">{label}</p>
    </div>
  );
}
