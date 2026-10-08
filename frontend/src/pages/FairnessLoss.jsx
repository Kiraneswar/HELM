// src/pages/FairnessLoss.jsx
import { useState, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import ChartWrapper from '../components/ChartWrapper';
import MethodToggle from '../components/MethodToggle';
import { CustomTooltip } from '../components/CustomTooltip';
import { METHOD_META, CHART_THEME } from '../constants';

function buildChartData(allResults, enabled) {
  if (!allResults?.length) return [];
  const roundMap = {};
  allResults.forEach(r => {
    if (!enabled.includes(r.method)) return;
    r.rounds.forEach(row => {
      if (!roundMap[row.fl_round]) roundMap[row.fl_round] = { round: row.fl_round };
      roundMap[row.fl_round][r.method + '_gini'] = row.gini_mean;
      roundMap[row.fl_round][r.method + '_loss'] = row.loss_mean;
    });
  });
  return Object.values(roundMap).sort((a, b) => a.round - b.round);
}

export default function FairnessLoss({ allResults, available }) {
  const [enabled, setEnabled] = useState(available);
  const chartData = useMemo(() => buildChartData(allResults, enabled), [allResults, enabled]);

  return (
    <div className="page-enter space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">Fairness & Loss</h2>
        <p className="text-sm text-white/40 mt-1">
          Gini coefficient (lower = fairer) and test loss vs. round
        </p>
      </div>

      <MethodToggle available={available} enabled={enabled} setEnabled={setEnabled} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <ChartWrapper title="Participation Fairness (Gini Coefficient)"
                      subtitle="0 = perfectly equal participation, 1 = monopoly">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
              <XAxis dataKey="round" stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }} />
              <YAxis stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                     domain={[0, 1]} tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip formatter={v => v.toFixed(4)} />} />
              {enabled.map(method => (
                <Line key={method} type="monotone" dataKey={method + '_gini'}
                      stroke={(METHOD_META[method] || {}).color || '#aaa'}
                      strokeWidth={method === 'federated_marl' ? 2.5 : 1.5}
                      dot={false}
                      name={(METHOD_META[method] || { label: method }).label} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartWrapper>

        <ChartWrapper title="Global Test Loss" subtitle="Cross-entropy; should decrease monotonically">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
              <XAxis dataKey="round" stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }} />
              <YAxis stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                     tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip formatter={v => v.toFixed(4)} />} />
              {enabled.map(method => (
                <Line key={method} type="monotone" dataKey={method + '_loss'}
                      stroke={(METHOD_META[method] || {}).color || '#aaa'}
                      strokeWidth={method === 'federated_marl' ? 2.5 : 1.5}
                      dot={false}
                      name={(METHOD_META[method] || { label: method }).label} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartWrapper>
      </div>
    </div>
  );
}
