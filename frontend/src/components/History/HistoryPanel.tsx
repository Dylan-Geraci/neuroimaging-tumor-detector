import { useEffect, useState } from 'react';
import { formatClassName } from '../../utils';

interface HistoryItem {
  id: string;
  created_at: string | null;
  filename: string;
  predicted_class: string;
  confidence: number;
  batch_id: string | null;
}

interface HistoryResponse {
  total: number;
  items: HistoryItem[];
}

export default function HistoryPanel() {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchHistory = async () => {
    try {
      const resp = await fetch('/predictions?limit=20');
      if (resp.ok) setData(await resp.json());
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (id: string) => {
    await fetch(`/predictions/${id}`, { method: 'DELETE' });
    fetchHistory();
  };

  if (loading) return <p className="loading-text">Loading history...</p>;
  if (!data || data.items.length === 0) return null;

  return (
    <div style={{ marginTop: '2.5rem' }}>
      <h3 style={{
        fontFamily: "'Source Serif 4', Georgia, serif",
        fontSize: '1.1rem',
        fontWeight: 600,
        marginBottom: '1rem',
        color: 'var(--text-primary)',
      }}>
        Recent Predictions ({data.total})
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        {data.items.map((item) => (
          <div
            key={item.id}
            className="scan-card"
            style={{ animation: 'none' }}
          >
            <div className="scan-details">
              <div className="scan-filename">{item.filename}</div>
              <div className="scan-prediction">{formatClassName(item.predicted_class)}</div>
              <div className="scan-confidence">{(item.confidence * 100).toFixed(1)}%</div>
              {item.created_at && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  {new Date(item.created_at).toLocaleString()}
                </div>
              )}
            </div>
            <button
              className="file-remove"
              onClick={() => handleDelete(item.id)}
              title="Delete"
            >
              &times;
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
