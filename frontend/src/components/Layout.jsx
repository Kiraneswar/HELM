// src/components/Layout.jsx
import { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, TrendingUp, Scale, Zap, Target, Brain, Table2, Settings, RefreshCw
} from 'lucide-react';

const NAV = [
  { to: '/',           icon: LayoutDashboard, label: 'Overview'        },
  { to: '/accuracy',   icon: TrendingUp,      label: 'Accuracy Curves' },
  { to: '/fairness',   icon: Scale,           label: 'Fairness & Loss' },
  { to: '/energy',     icon: Zap,             label: 'Energy Cost'     },
  { to: '/tradeoff',   icon: Target,          label: 'Tradeoff'        },
  { to: '/rl',         icon: Brain,           label: 'RL Diagnostics'  },
  { to: '/table',      icon: Table2,          label: 'Comparison Table'},
];

export default function Layout({ children, resultsDir, setResultsDir, onReload, loading }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft]     = useState(resultsDir);

  function submitDir(e) {
    e.preventDefault();
    setResultsDir(draft);
    setEditing(false);
  }

  return (
    <div className="flex min-h-screen bg-[#03001C]">
      {/* ── Sidebar ── */}
      <aside className="w-64 shrink-0 flex flex-col glass border-r border-white/8 p-4">
        {/* Logo */}
        <div className="mb-8 mt-2 px-2">
          <div className="text-xs font-mono text-violet-400 tracking-widest mb-1">RESEARCH DASHBOARD</div>
          <h1 className="text-lg font-bold text-white leading-tight">
            Federated<br/>
            <span className="text-violet-400">Multi-Agent RL</span>
          </h1>
          <div className="mt-2 inline-block bg-gradient-to-r from-pink-500 to-violet-600 text-white
                          text-[10px] font-bold px-2 py-0.5 rounded-full tracking-wider">
            NOVEL CONTRIBUTION
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 space-y-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-violet-600/30 text-violet-300 border border-violet-500/40'
                    : 'text-white/50 hover:text-white/80 hover:bg-white/5'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Results dir control */}
        <div className="mt-6 pt-4 border-t border-white/8">
          <div className="text-xs text-white/40 mb-2 flex items-center gap-1">
            <Settings size={11}/> Results directory
          </div>
          {editing ? (
            <form onSubmit={submitDir} className="flex gap-1">
              <input
                className="flex-1 bg-white/5 border border-white/15 rounded px-2 py-1
                           text-xs text-white font-mono outline-none focus:border-violet-400"
                value={draft}
                onChange={e => setDraft(e.target.value)}
                autoFocus
              />
              <button type="submit"
                className="bg-violet-600 hover:bg-violet-500 text-white text-xs px-2 py-1 rounded">
                OK
              </button>
            </form>
          ) : (
            <button
              onClick={() => { setEditing(true); setDraft(resultsDir); }}
              className="w-full text-left text-xs text-white/50 hover:text-white/80
                         font-mono truncate bg-white/3 rounded px-2 py-1 border border-white/8">
              {resultsDir || '(click to set)'}
            </button>
          )}

          <button
            onClick={onReload}
            disabled={loading}
            className="mt-2 w-full flex items-center justify-center gap-2 text-xs
                       text-violet-400 hover:text-violet-300 bg-violet-600/10
                       hover:bg-violet-600/20 rounded-lg py-1.5 transition-all border border-violet-500/20">
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            Refresh data
          </button>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 overflow-auto p-6 page-enter">
        {children}
      </main>
    </div>
  );
}
