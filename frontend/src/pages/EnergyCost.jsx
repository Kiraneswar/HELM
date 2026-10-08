// src/pages/EnergyCost.jsx
import { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import ChartWrapper from '../components/ChartWrapper';
import MethodToggle from '../components/MethodToggle';
import { CustomTooltip } from '../components/CustomTooltip';
import { METHOD_META, CHART_THEME } from '../constants';

export default function EnergyCost({ allResults, available }) {
  const [enabled, setEnabled] = useState(available);

  const chartData = useMemo(() => {
    if (!allResults?.length) return [];
    const roundMap = {};
    allResults.forEach(r => {
      if (!enabled.includes(r.method)) return;
      r.rounds.forEach(row => {
        if (!roundMap[row.fl_round]) roundMap[row.fl_round] = { round: row.fl_round };
        roundMap[row.fl_round][r.method] = row.energy_mean;
      });
    });
    return Object.values(roundMap).sort((a, b) => a.round - b.round);
  }, [allResults, enabled]);

  return (
    <div className="page-enter space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">Energy Cost per Round</h2>
        <p className="text-sm text-white/40 mt-1">
          Average total energy consumed per FL round (arbitrary units; lower = more efficient)
        </p>
      </div>

      <MethodToggle available={available} enabled={enabled} setEnabled={setEnabled} />

      <ChartWrapper title="Energy Cost vs. FL Round" subtitle="Grouped bars — one per enabled method">
        <ResponsiveContainer width="100%" height={380}>
          <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
            <XAxis dataKey="round" stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 10 }} />
            <YAxis stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }} />
            <Tooltip content={<CustomTooltip formatter={v => v.toFixed(4)} />} />
            {enabled.map(method => (
              <Bar key={method} dataKey={method}
                   fill={(METHOD_META[method] || {}).color || '#aaa'}
                   fillOpacity={method === 'federated_marl' ? 0.9 : 0.6}
                   name={(METHOD_META[method] || { label: method }).label}
                   radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </div>
  );
}
