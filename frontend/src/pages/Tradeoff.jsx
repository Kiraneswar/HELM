// src/pages/Tradeoff.jsx — Accuracy vs. Energy scatter
import { useMemo } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Label,
} from 'recharts';
import ChartWrapper from '../components/ChartWrapper';
import { METHOD_META, CHART_THEME, NOVEL_METHOD } from '../constants';

function ScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: CHART_THEME.tooltipBg,
      border: `1px solid ${CHART_THEME.tooltipBorder}`,
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: d.color, fontWeight: 600, marginBottom: 4 }}>{d.label}</div>
      <div style={{ color: 'rgba(255,255,255,0.7)' }}>Accuracy: <b>{(d.accuracy * 100).toFixed(1)}%</b></div>
      <div style={{ color: 'rgba(255,255,255,0.7)' }}>Energy: <b>{d.energy.toFixed(3)}</b></div>
      <div style={{ color: 'rgba(255,255,255,0.7)' }}>Gini: <b>{d.gini.toFixed(3)}</b></div>
    </div>
  );
}

export default function Tradeoff({ summary }) {
  const points = useMemo(() => {
    if (!summary) return [];
    return summary.map(card => ({
      method:   card.method,
      label:    card.label || METHOD_META[card.method]?.label || card.method,
      color:    METHOD_META[card.method]?.color || '#aaa',
      accuracy: card.final_accuracy_mean,
      energy:   card.total_energy,
      gini:     card.final_gini,
      isNovel:  card.method === NOVEL_METHOD,
    }));
  }, [summary]);

  const renderDot = (props) => {
    const { cx, cy, payload } = props;
    return (
      <g key={payload.method}>
        <circle cx={cx} cy={cy} r={payload.isNovel ? 9 : 6}
                fill={payload.color}
                fillOpacity={0.85}
                stroke={payload.isNovel ? '#fff' : 'none'}
                strokeWidth={1.5} />
        <text x={cx} y={cy - 12} textAnchor="middle"
              fill="rgba(255,255,255,0.7)" fontSize={10}>
          {payload.label.split(' ')[0]}
        </text>
      </g>
    );
  };

  return (
    <div className="page-enter space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">Accuracy vs. Energy Tradeoff</h2>
        <p className="text-sm text-white/40 mt-1">
          One point per method — upper-left is ideal (high accuracy, low energy)
        </p>
      </div>

      <ChartWrapper title="Final Accuracy vs. Total Energy (all rounds)"
                    subtitle="Dot size: novel method (Federated MARL) is larger">
        <ResponsiveContainer width="100%" height={420}>
          <ScatterChart margin={{ top: 24, right: 40, left: 8, bottom: 36 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
            <XAxis type="number" dataKey="energy" name="Energy"
                   stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}>
              <Label value="Total Energy (all rounds)" position="insideBottom"
                     offset={-20} fill="rgba(255,255,255,0.35)" fontSize={11} />
            </XAxis>
            <YAxis type="number" dataKey="accuracy" name="Accuracy"
                   tickFormatter={v => `${(v*100).toFixed(0)}%`}
                   stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}>
              <Label value="Final Accuracy" angle={-90} position="insideLeft"
                     offset={10} fill="rgba(255,255,255,0.35)" fontSize={11} />
            </YAxis>
            <Tooltip content={<ScatterTooltip />} />
            <Scatter data={points} shape={renderDot} />
          </ScatterChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </div>
  );
}
