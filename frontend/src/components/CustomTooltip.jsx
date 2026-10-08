// src/components/CustomTooltip.jsx
// Shared Recharts tooltip component for all line/bar charts.
import { CHART_THEME } from '../constants';

export function CustomTooltip({ active, payload, label, formatter }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: CHART_THEME.tooltipBg,
      border: `1px solid ${CHART_THEME.tooltipBorder}`,
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
      fontFamily: 'Inter, sans-serif',
    }}>
      <div style={{ color: 'rgba(255,255,255,0.6)', marginBottom: 6 }}>
        Round {label}
      </div>
      {payload.map((entry, i) => (
        <div key={i} style={{ color: entry.color, marginBottom: 2 }}>
          <span style={{ fontWeight: 500 }}>{entry.name}:</span>{' '}
          <span className="font-mono">
            {formatter ? formatter(entry.value, entry.name) : entry.value.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  );
}
