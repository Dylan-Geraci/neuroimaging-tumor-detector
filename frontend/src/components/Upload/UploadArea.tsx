import { useRef, useState, useCallback } from 'react';

interface Props {
  onFiles: (files: FileList | File[]) => void;
}

export default function UploadArea({ onFiles }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragover, setDragover] = useState(false);

  const handleClick = () => inputRef.current?.click();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFiles(e.target.files);
    }
    e.target.value = '';
  };

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragover(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragover(false), []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragover(false);
      const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'));
      if (files.length > 0) onFiles(files);
    },
    [onFiles],
  );

  return (
    <div
      className={`upload-area${dragover ? ' dragover' : ''}`}
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="upload-icon-circle" aria-hidden="true">
        +
      </div>
      <h2>Upload MRI Scans</h2>
      <p>Drag and drop images here, or click to select</p>
      <p className="file-info">Supported formats: JPG, PNG | Hold Cmd and click to select multiple files</p>
      <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={handleChange} />
    </div>
  );
}
