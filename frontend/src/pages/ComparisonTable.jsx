// src/pages/ComparisonTable.jsx
import { useMemo } from 'react';
import { METHOD_META, NOVEL_METHOD } from '../constants';

function fmt(val, digits = 3) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  return val.toFixed(digits);
}
function pct(val, digits = 1) {
  if (val === null || val === undefined || isNaN(val)) return '—';
  return `${(val * 100).toFixed(digits)}%`;
}

export default function ComparisonTable({ summary, allResults }) {
  const rows = useMemo(() => {
    if (!summary) return [];
    return summary.map(card => {
      const result = allResults?.find(r => r.method === card.method);
      const firstRound = result?.rounds?.[0];
      const peakAcc = result
        ? Math.max(...result.rounds.map(r => r.accuracy_mean))
        : null;
      return {
        method:       card.method,
        label:        METHOD_META[card.method]?.label || card.method,
        color:        METHOD_META[card.method]?.color || '#aaa',
        isNovel:      card.method === NOVEL_METHOD,
        finalAcc:     card.final_accuracy_mean,
        finalAccStd:  card.final_accuracy_std,
        peakAcc,
        totalEnergy:  card.total_energy,
        finalGini:    card.final_gini,
        finalGiniStd: card.final_gini_std,
        numSeeds:     card.num_seeds,
        numRounds:    card.num_rounds,
      };
    }).sort((a, b) => {
      if (a.isNovel) return -1;
      if (b.isNovel) return  1;
      return b.finalAcc - a.finalAcc;
    });
  }, [summary, allResults]);

  const cols = [
    { header: 'Method',           render: r => (
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color }} />
        <span style={{ color: r.isNovel ? r.color : undefined }}>
          {r.label}
          {r.isNovel && <span className="ml-2 text-[9px] bg-pink-500/20 text-pink-300 px-1.5 rounded-full">OURS</span>}
        </span>
      </div>
    )},
    { header: 'Final Acc (%)',     render: r => (
      <span className="font-mono">{pct(r.finalAcc)} <span className="text-white/30 text-xs">±{pct(r.finalAccStd)}</span></span>
    )},
    { header: 'Peak Acc (%)',      render: r => <span className="font-mono">{pct(r.peakAcc)}</span> },
    { header: 'Total Energy',      render: r => <span className="font-mono">{fmt(r.totalEnergy, 2)}</span> },
    { header: 'Final Gini',        render: r => (
      <span className="font-mono">{fmt(r.finalGini)} <span className="text-white/30 text-xs">±{fmt(r.finalGiniStd)}</span></span>
    )},
    { header: 'Seeds',             render: r => <span className="font-mono">{r.numSeeds}</span> },
    { header: 'Rounds',            render: r => <span className="font-mono">{r.numRounds}</span> },
  ];

  return (
    <div className="page-enter">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white">Comparison Table</h2>
        <p className="text-sm text-white/40 mt-1">
          Final-round metrics — mean ± std across seeds. Sorted: our method first, then by accuracy.
        </p>
      </div>

      <div className="glass rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-white/40 text-xs uppercase tracking-wider border-b border-white/10">
                {cols.map(c => (
                  <th key={c.header} className="px-4 py-3 text-left font-medium">{c.header}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {rows.map(r => (
                <tr key={r.method}
                    className={`hover:bg-white/3 transition-colors duration-100 ${
                      r.isNovel ? 'bg-pink-500/5' : ''
                    }`}>
                  {cols.map(c => (
                    <td key={c.header} className="px-4 py-3 text-white/80 whitespace-nowrap">
                      {c.render(r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="px-4 py-3 border-t border-white/8 text-xs text-white/30">
          Gini = participation fairness coefficient (lower is better). Energy = sum of all round energy costs.
          Std shown where num_seeds &gt; 1.
        </div>
      </div>
    </div>
  );
}
