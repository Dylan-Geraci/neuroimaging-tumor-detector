import type { PredictionImages } from '../../api/types';

interface Props {
  images: PredictionImages;
}

export default function GradCamViewer({ images }: Props) {
  return (
    <div className="visualizations">
      <div className="viz-item">
        <h3>Original MRI</h3>
        <img src={images.original} alt="Original MRI" />
      </div>
      <div className="viz-item">
        <h3>Attention Heatmap</h3>
        <img src={images.heatmap} alt="Grad-CAM Heatmap" />
      </div>
      <div className="viz-item">
        <h3>Overlay</h3>
        <img src={images.overlay} alt="Overlay" />
      </div>
    </div>
  );
}
