// src/pages/RLDiagnostics.jsx
// PPO training diagnostics for federated_marl only.
import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import ChartWrapper from '../components/ChartWrapper';
import { CustomTooltip } from '../components/CustomTooltip';
import { CHART_THEME } from '../constants';

export default function RLDiagnostics({ allResults }) {
  const data = useMemo(() => {
    const marl = allResults?.find(r => r.method === 'federated_marl');
    if (!marl) return [];
    return marl.rounds.map(row => ({
      round:        row.fl_round,
      policy_loss:  row.policy_loss_mean,
      entropy:      row.entropy_mean,
      policy_sim:   row.policy_sim_mean,
    }));
  }, [allResults]);

  if (!data.length) {
    return (
      <div className="page-enter">
        <h2 className="text-xl font-bold text-white mb-2">RL Diagnostics</h2>
        <div className="glass rounded-xl p-8 text-center text-white/40 text-sm">
          No federated_marl results available in the selected results directory.
        </div>
      </div>
    );
  }

  const lines = [
    { key: 'policy_loss', label: 'PPO Policy Loss',           color: '#E91E63',
      note: 'Should decrease / stabilise — confirms RL agent is converging' },
    { key: 'entropy',     label: 'Policy Entropy',            color: '#FF9800',
      note: 'High early (exploration), low late (exploitation)' },
    { key: 'policy_sim',  label: 'Cross-Cluster Similarity',  color: '#2DD4BF',
      note: 'Rises after each T_RL federation step (policies converge globally)' },
  ];

  return (
    <div className="page-enter space-y-4">
      <div className="mb-4">
        <h2 className="text-xl font-bold text-white">RL Training Diagnostics</h2>
        <p className="text-sm text-white/40 mt-1">
          Federated MARL (Ours) — PPO convergence + cross-cluster policy similarity
        </p>
      </div>

      <div className="glass rounded-xl p-4 mb-2 border border-pink-500/20">
        <div className="flex items-center gap-2 mb-1">
          <span className="w-1 h-3 bg-pink-500 rounded-full" />
          <span className="text-xs font-semibold text-pink-300">Novel Contribution</span>
        </div>
        <p className="text-xs text-white/50 leading-relaxed">
          The cross-cluster policy similarity metric is unique to this paper: it measures
          cosine similarity between per-cluster ActorCriticPolicy weights after each T_RL
          federated averaging step. A rising trend confirms that federation is genuinely
          transferring RL knowledge between clusters — not just averaging random noise.
        </p>
      </div>

      {lines.map(({ key, label, color, note }) => (
        <ChartWrapper key={key} title={label} subtitle={note}>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={CHART_THEME.gridColor} />
              <XAxis dataKey="round" stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }} />
              <YAxis stroke={CHART_THEME.axisColor}
                     tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 11 }} />
              <Tooltip content={<CustomTooltip formatter={v => v.toFixed(5)} />} />
              <Line type="monotone" dataKey={key} stroke={color}
                    strokeWidth={2.5} dot={false} name={label} />
            </LineChart>
          </ResponsiveContainer>
        </ChartWrapper>
      ))}
    </div>
  );
}
