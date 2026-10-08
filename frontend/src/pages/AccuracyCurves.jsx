// src/pages/AccuracyCurves.jsx
import { useState, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import ChartWrapper from '../components/ChartWrapper';
import MethodToggle from '../components/MethodToggle';
import { CustomTooltip } from '../components/CustomTooltip';
import { METHOD_META, CHART_THEME } from '../constants';

export default function AccuracyCurves({ allResults, available }) {
  const [enabled, setEnabled] = useState(available);
  const [range,   setRange]   = useState([1, null]);  // null = all rounds

  const maxRound = useMemo(() => {
    if (!allResults?.length) return 100;
    return Math.max(...allResults.map(r => r.rounds.at(-1)?.fl_round ?? 0));
  }, [allResults]);

  // Build one merged data array keyed by fl_round
  const chartData = useMemo(() => {
    if (!allResults?.length) return [];
    const roundMap = {};
    allResults.forEach(result => {
      if (!enabled.includes(result.method)) return;
      result.rounds.forEach(row => {
        if (!roundMap[row.fl_round]) roundMap[row.fl_round] = { round: row.fl_round };
        roundMap[row.fl_round][result.method]        = row.accuracy_mean * 100;
        roundMap[row.fl_round][result.method + '_hi'] = (row.accuracy_mean + row.accuracy_std) * 100;
        roundMap[row.fl_round][result.method + '_lo'] = (row.accuracy_mean - row.accuracy_std) * 100;
      });
    });
    return Object.values(roundMap).sort((a, b) => a.round - b.round)
      .filter(r => r.round >= (range[0] ?? 1) && r.round <= (range[1] ?? maxRound));
  }, [allResults, enabled, range, maxRound]);

  return (
    <div className="page-enter space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">Accuracy Curves</h2>
        <p className="text-sm text-white/40 mt-1">Global FL model accuracy vs. round, all methods</p>
      </div>

      <ChartWrapper
        title="Test Accuracy (%)"
        subtitle="Solid line = mean across seeds; shaded band = ±1 std"
      >
        {/* Method toggles */}
        <MethodToggle available={available} enabled={enabled} setEnabled={setEnabled} />

        {/* Round range slider */}
        <div className="flex items-center gap-3 mb-4 text-xs text-white/50">
          <span>Round range:</span>
          <input type="number" min={1} max={maxRound}
            className="w-16 bg-white/5 border border-white/12 rounded px-2 py-0.5 text-white font-mono text-xs"
            value={range[0]}
            onChange={e => setRange([+e.target.value, range[1]])}
          />
          <span>–</span>
          <input type="number" min={1} max={maxRound}
            className="w-16 bg-white/5 border border-white/12 rounded px-2 py-0.5 text-white font-mono text-xs"
            value={range[1] ?? maxRound}
            onChange={e => setRange([range[0], +e.target.value])}
          />
        </div>

        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
            <XAxis dataKey="round" stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                   label={{ value: 'FL Round', position: 'insideBottom', offset: -2,
                            fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
            <YAxis stroke={CHART_THEME.axisColor}
                   tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }}
                   domain={[0, 100]}
                   tickFormatter={v => `${v}%`} />
            <ReferenceLine y={10} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4"
                           label={{ value: 'Random chance (10%)', fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
            <Tooltip content={
              <CustomTooltip formatter={(v, n) => `${v.toFixed(1)}%`} />
            } />
            {enabled.map(method => {
              const meta = METHOD_META[method] || { color: '#aaa', label: method };
              return (
                <Line
                  key={method}
                  type="monotone"
                  dataKey={method}
                  stroke={meta.color}
                  strokeWidth={method === 'federated_marl' ? 3 : 1.5}
                  dot={false}
                  name={meta.label}
                  strokeDasharray={method === 'full' ? '6 3' : undefined}
                />
              );
            })}
          </LineChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </div>
  );
}
