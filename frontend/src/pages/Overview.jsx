// src/pages/Overview.jsx
// Final-round summary cards for all methods + system architecture panel.
import { METHOD_META, NOVEL_METHOD } from '../constants';

function MetricCard({ card }) {
  const meta    = METHOD_META[card.method] || { label: card.method, color: '#aaa', isNovel: false };
  const isNovel = card.method === NOVEL_METHOD;
  const acc     = (card.final_accuracy_mean * 100).toFixed(1);
  const std     = (card.final_accuracy_std  * 100).toFixed(1);

  return (
    <div
      className={`glass rounded-xl p-5 transition-all duration-200 hover:-translate-y-0.5 ${
        isNovel ? 'glow-pink border-pink-500/30' : 'hover:border-white/20'
      }`}
    >
      {isNovel && (
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[10px] font-bold bg-gradient-to-r from-pink-500 to-violet-600
                           text-white px-2 py-0.5 rounded-full tracking-wider">
            ★ OUR METHOD
          </span>
        </div>
      )}
      <div className="text-xs font-medium text-white/50 mb-1 uppercase tracking-wider">
        {meta.label}
      </div>
      <div className="text-3xl font-bold mb-1" style={{ color: meta.color }}>
        {acc}%
        <span className="text-sm font-normal text-white/30 ml-1">±{std}%</span>
      </div>
      <div className="h-px bg-white/8 my-3" />
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <div className="text-white/35 mb-0.5">Total Energy</div>
          <div className="font-mono text-white/70">{card.total_energy.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-white/35 mb-0.5">Gini (fairness)</div>
          <div className="font-mono text-white/70">{card.final_gini.toFixed(3)}</div>
        </div>
        <div>
          <div className="text-white/35 mb-0.5">Seeds</div>
          <div className="font-mono text-white/70">{card.num_seeds}</div>
        </div>
        <div>
          <div className="text-white/35 mb-0.5">Rounds</div>
          <div className="font-mono text-white/70">{card.num_rounds}</div>
        </div>
      </div>
    </div>
  );
}

export default function Overview({ summary }) {
  if (!summary?.length) {
    return <div className="text-white/40 text-sm">No data loaded — check results directory.</div>;
  }

  // Sort: novel method first, then others by final accuracy descending
  const sorted = [...summary].sort((a, b) => {
    if (a.method === NOVEL_METHOD) return -1;
    if (b.method === NOVEL_METHOD) return  1;
    return b.final_accuracy_mean - a.final_accuracy_mean;
  });

  return (
    <div className="page-enter">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white">Overview</h2>
        <p className="text-sm text-white/40 mt-1">
          Final-round summary across all methods — {sorted[0]?.num_rounds || '?'} FL rounds
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 mb-8">
        {sorted.map(card => <MetricCard key={card.method} card={card} />)}
      </div>

      {/* Architecture panel */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-sm font-semibold text-white/90 mb-4 flex items-center gap-2">
          <span className="w-1 h-4 bg-violet-500 rounded-full inline-block" />
          System Architecture & Design Decisions
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="text-white/40 border-b border-white/8">
                <th className="pb-2 pr-6 font-medium">Layer</th>
                <th className="pb-2 pr-6 font-medium">Actor</th>
                <th className="pb-2 pr-6 font-medium">What federates</th>
                <th className="pb-2 font-medium">Cadence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              <tr className="text-white/70">
                <td className="py-2 pr-6 font-mono">Edge tier</td>
                <td className="py-2 pr-6">Per-cluster ActorCriticPolicy</td>
                <td className="py-2 pr-6">Selects k clients/round</td>
                <td className="py-2 font-mono text-teal-400">every round</td>
              </tr>
              <tr className="text-white/70">
                <td className="py-2 pr-6 font-mono">Cloud tier</td>
                <td className="py-2 pr-6">FedAvgAggregator</td>
                <td className="py-2 pr-6">Averages FL model weights</td>
                <td className="py-2 font-mono text-violet-400">every T_FL rounds</td>
              </tr>
              <tr className="text-pink-300">
                <td className="py-2 pr-6 font-mono">Cloud tier</td>
                <td className="py-2 pr-6 font-medium">FedAvgAggregator</td>
                <td className="py-2 pr-6 font-medium">
                  Averages RL policy weights
                  <span className="ml-2 text-[9px] bg-pink-500/30 text-pink-300 px-1 rounded">NOVEL</span>
                </td>
                <td className="py-2 font-mono text-pink-400">every T_RL rounds</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-4 pt-4 border-t border-white/8 text-xs text-white/35 space-y-1.5">
          <p><span className="text-white/55 font-medium">Why pure PyTorch (no Flower)?</span>{' '}
            Flower's round structure doesn't support independent T_FL/T_RL cadences plus
            in-round SimPy churn events. Pure PyTorch gives the runner full control over
            round timing and SimPy event injection.</p>
          <p><span className="text-white/55 font-medium">Why PPO over REINFORCE?</span>{' '}
            REINFORCE's high-variance gradient estimates caused unstable policies.
            PPO's clipped objective with GAE keeps updates conservative.</p>
          <p><span className="text-white/55 font-medium">Why sequential multinomial for top-k?</span>{' '}
            Gives exact discrete selections with well-defined log-probabilities for PPO's
            importance ratio. Gumbel-softmax produces "soft" selections harder to log.</p>
        </div>
      </div>
    </div>
  );
}
