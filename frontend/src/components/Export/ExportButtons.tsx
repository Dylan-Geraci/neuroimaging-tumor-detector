import type { PredictionResponse } from '../../api/types';
import { isBatchResponse } from '../../api/types';
import { formatClassName, getTimestamp, formatDate, downloadFile } from '../../utils';
import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';

interface Props {
  result: PredictionResponse;
}

const CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary'] as const;

function exportToCSV(result: PredictionResponse) {
  const headerLabels = CLASS_NAMES.map((c) => `Probability_${formatClassName(c)}`);
  const rows: string[] = [];

  if (isBatchResponse(result)) {
    const { aggregated_prediction, individual_predictions } = result;
    rows.push(['Filename', 'Classification', 'Confidence', ...headerLabels].join(','));

    individual_predictions.forEach((pred) => {
      if (pred.error) {
        rows.push([pred.filename, 'Error', pred.error, '', '', '', ''].join(','));
        return;
      }
      const probs = CLASS_NAMES.map((c) => (pred.probabilities[c] * 100).toFixed(2) + '%');
      rows.push(
        [pred.filename, formatClassName(pred.class), (pred.confidence * 100).toFixed(2) + '%', ...probs].join(','),
      );
    });

    rows.push('');
    rows.push('Aggregated Result');
    const aggProbs = CLASS_NAMES.map((c) => (aggregated_prediction.probabilities[c] * 100).toFixed(2) + '%');
    rows.push(
      [
        'AGGREGATED',
        formatClassName(aggregated_prediction.class),
        (aggregated_prediction.confidence * 100).toFixed(2) + '%',
        ...aggProbs,
      ].join(','),
    );
    rows.push(`Agreement Score,${(aggregated_prediction.agreement_score * 100).toFixed(0)}%`);
  } else {
    const { prediction } = result;
    rows.push(['Timestamp', 'Classification', 'Confidence', ...headerLabels].join(','));
    const probs = CLASS_NAMES.map((c) => (prediction.probabilities[c] * 100).toFixed(2) + '%');
    rows.push(
      [formatDate(new Date()), formatClassName(prediction.class), (prediction.confidence * 100).toFixed(2) + '%', ...probs].join(','),
    );
  }

  downloadFile(rows.join('\n'), `brain-tumor-results-${getTimestamp()}.csv`, 'text/csv');
}

function exportToPDF(result: PredictionResponse) {
  const doc = new jsPDF('p', 'mm', 'a4');
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 20;
  const contentWidth = pageWidth - margin * 2;
  let y = 0;

  const TERRACOTTA: [number, number, number] = [196, 112, 75];
  const TEXT_PRIMARY: [number, number, number] = [46, 42, 37];
  const TEXT_SECONDARY: [number, number, number] = [120, 113, 108];
  const STONE_100: [number, number, number] = [240, 238, 235];

  const batch = isBatchResponse(result);

  function checkPageBreak(needed: number) {
    if (y + needed > pageHeight - 20) {
      doc.addPage();
      y = margin;
    }
  }

  // Header bar
  doc.setFillColor(...TERRACOTTA);
  doc.rect(0, 0, pageWidth, 36, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(18);
  doc.setFont('helvetica', 'bold');
  doc.text('Brain Tumor Classification Report', margin, 16);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.text(formatDate(new Date()), margin, 24);
  const subtitle = batch ? `Batch Analysis \u2014 ${result.processed_count} scans` : 'Single Scan Analysis';
  doc.text(subtitle, margin, 30);
  y = 46;

  // Diagnosis
  if (batch) {
    const agg = result.aggregated_prediction;
    doc.setFontSize(9);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.setFont('helvetica', 'normal');
    doc.text('AGGREGATED DIAGNOSIS', margin, y);
    y += 8;
    doc.setFontSize(22);
    doc.setTextColor(...TEXT_PRIMARY);
    doc.setFont('helvetica', 'bold');
    doc.text(formatClassName(agg.class), margin, y);
    y += 8;
    doc.setFontSize(11);
    doc.setTextColor(...TERRACOTTA);
    doc.setFont('helvetica', 'normal');
    doc.text(`${(agg.confidence * 100).toFixed(2)}% Confidence`, margin, y);
    doc.text(`${(agg.agreement_score * 100).toFixed(0)}% Agreement`, margin + 60, y);
    y += 12;
  } else {
    const pred = result.prediction;
    doc.setFontSize(9);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.setFont('helvetica', 'normal');
    doc.text('DIAGNOSIS', margin, y);
    y += 8;
    doc.setFontSize(22);
    doc.setTextColor(...TEXT_PRIMARY);
    doc.setFont('helvetica', 'bold');
    doc.text(formatClassName(pred.class), margin, y);
    y += 8;
    doc.setFontSize(11);
    doc.setTextColor(...TERRACOTTA);
    doc.setFont('helvetica', 'normal');
    doc.text(`${(pred.confidence * 100).toFixed(2)}% Confidence`, margin, y);
    y += 12;
  }

  // Probability bars
  const probs = batch ? result.aggregated_prediction.probabilities : result.prediction.probabilities;
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  doc.setFontSize(9);
  doc.setTextColor(...TEXT_SECONDARY);
  doc.text('CLASS PROBABILITIES', margin, y);
  y += 7;

  sorted.forEach(([cls, prob], index) => {
    checkPageBreak(12);
    const barHeight = 5;
    const barMaxWidth = contentWidth - 50;

    doc.setFontSize(9);
    doc.setTextColor(...TEXT_PRIMARY);
    doc.setFont('helvetica', 'normal');
    doc.text(formatClassName(cls), margin, y);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.text(`${(prob * 100).toFixed(2)}%`, pageWidth - margin, y, { align: 'right' });
    y += 2;

    doc.setFillColor(...STONE_100);
    doc.roundedRect(margin, y, barMaxWidth, barHeight, 1, 1, 'F');
    const fillColor: [number, number, number] = index === 0 ? TERRACOTTA : [200, 195, 188];
    doc.setFillColor(...fillColor);
    doc.roundedRect(margin, y, Math.max(1, barMaxWidth * prob), barHeight, 1, 1, 'F');
    y += barHeight + 5;
  });

  y += 5;

  // Batch table or single images
  if (batch) {
    checkPageBreak(30);
    doc.setFontSize(9);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.text('INDIVIDUAL SCAN RESULTS', margin, y);
    y += 4;

    const tableData = result.individual_predictions.map((pred) => {
      if (pred.error) return [pred.filename, 'Error', pred.error];
      return [pred.filename, formatClassName(pred.class), (pred.confidence * 100).toFixed(2) + '%'];
    });

    autoTable(doc, {
      startY: y,
      margin: { left: margin, right: margin },
      head: [['Filename', 'Classification', 'Confidence']],
      body: tableData,
      theme: 'grid',
      headStyles: { fillColor: TERRACOTTA, textColor: [255, 255, 255], fontSize: 8, fontStyle: 'bold' },
      bodyStyles: { fontSize: 8, textColor: TEXT_PRIMARY },
      alternateRowStyles: { fillColor: [250, 249, 247] },
      styles: { cellPadding: 3, lineColor: [226, 223, 218], lineWidth: 0.25 },
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    y = (doc as any).lastAutoTable.finalY + 10;

    // Sample overlay images
    const valid = result.individual_predictions.filter((p) => !p.error && p.images?.overlay);
    const sampled = valid.slice(0, 6);
    if (sampled.length > 0) {
      checkPageBreak(50);
      doc.setFontSize(9);
      doc.setTextColor(...TEXT_SECONDARY);
      doc.text('SAMPLE OVERLAY IMAGES', margin, y);
      y += 5;
      const cols = 3;
      const imgSize = (contentWidth - (cols - 1) * 5) / cols;

      for (let i = 0; i < sampled.length; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        if (col === 0 && row > 0) {
          y += imgSize + 12;
          checkPageBreak(imgSize + 12);
        }
        const x = margin + col * (imgSize + 5);
        try {
          doc.addImage(sampled[i].images.overlay, 'JPEG', x, y, imgSize, imgSize);
          doc.setFontSize(6);
          doc.setTextColor(...TEXT_SECONDARY);
          doc.text(sampled[i].filename, x, y + imgSize + 3, { maxWidth: imgSize });
        } catch {
          // skip
        }
      }
      y += imgSize + 12;
    }
  } else {
    const { images } = result;
    checkPageBreak(70);
    doc.setFontSize(9);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.text('GRAD-CAM VISUALIZATIONS', margin, y);
    y += 5;
    const imgSize = (contentWidth - 10) / 3;
    const labels = ['Original MRI', 'Attention Heatmap', 'Overlay'];
    const srcs = [images.original, images.heatmap, images.overlay];

    for (let i = 0; i < srcs.length; i++) {
      const x = margin + i * (imgSize + 5);
      try {
        doc.addImage(srcs[i], 'JPEG', x, y, imgSize, imgSize);
        doc.setFontSize(7);
        doc.setTextColor(...TEXT_SECONDARY);
        doc.text(labels[i], x + imgSize / 2, y + imgSize + 4, { align: 'center' });
      } catch {
        // skip
      }
    }
  }

  // Footers
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const totalPages = (doc as any).internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFontSize(7);
    doc.setTextColor(...TEXT_SECONDARY);
    doc.text(
      'This tool is not intended for clinical diagnosis. ResNet18 model \u00B7 97.10% test accuracy.',
      margin,
      pageHeight - 10,
    );
    doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 10, { align: 'right' });
  }

  doc.save(`brain-tumor-report-${getTimestamp()}.pdf`);
}

export default function ExportButtons({ result }: Props) {
  return (
    <div className="export-section">
      <button className="export-btn" onClick={() => exportToCSV(result)}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
        </svg>
        Export CSV
      </button>
      <button className="export-btn" onClick={() => exportToPDF(result)}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="12" y1="18" x2="12" y2="12" />
          <polyline points="9 15 12 18 15 15" />
        </svg>
        Export PDF
      </button>
    </div>
  );
}
