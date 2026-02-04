import { useState, useCallback } from 'react';

const IMAGE_EXTENSIONS = /\.(jpg|jpeg|png|gif|bmp|webp)$/i;

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_EXTENSIONS.test(file.name);
}

export function useFileSelection() {
  const [files, setFiles] = useState<File[]>([]);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const valid = Array.from(incoming).filter(isImageFile);
    if (valid.length > 0) {
      setFiles((prev) => [...prev, ...valid]);
    }
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const clearFiles = useCallback(() => {
    setFiles([]);
  }, []);

  return { files, addFiles, removeFile, clearFiles };
}
