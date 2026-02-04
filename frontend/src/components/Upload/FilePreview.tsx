import { useMemo } from 'react';

interface Props {
  files: File[];
  onRemove: (index: number) => void;
  onClear: () => void;
  onUpload: () => void;
  onAddMore: () => void;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function FilePreview({ files, onRemove, onClear, onUpload, onAddMore }: Props) {
  const thumbnails = useMemo(
    () => files.map((f) => URL.createObjectURL(f)),
    [files],
  );

  if (files.length === 0) return null;

  return (
    <div className="file-preview">
      <div className="file-preview-header">
        <h3>
          Selected Files: <span className="file-count-badge">{files.length}</span>
        </h3>
        <button className="add-more-btn" onClick={onAddMore}>
          + Add More
        </button>
      </div>

      <div className="file-list">
        {files.map((file, index) => (
          <div
            key={`${file.name}-${index}`}
            className="file-item"
            style={{ animation: `fadeUp 0.3s cubic-bezier(0.22, 1, 0.36, 1) ${index * 0.03}s both` }}
          >
            <div className="file-thumbnail">
              <img src={thumbnails[index]} alt={file.name} />
            </div>
            <div className="file-item-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">{formatFileSize(file.size)}</span>
            </div>
            <button className="file-remove" onClick={() => onRemove(index)}>
              &times;
            </button>
          </div>
        ))}
      </div>

      <div className="file-actions">
        <button className="clear-btn" onClick={onClear}>
          Clear Selection
        </button>
        <button className="upload-btn" onClick={onUpload}>
          {files.length === 1 ? 'Analyze Scan' : `Analyze ${files.length} Scans`}
        </button>
      </div>
    </div>
  );
}
