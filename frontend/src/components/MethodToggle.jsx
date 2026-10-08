// src/components/MethodToggle.jsx
// Checkbox toggles for which methods to show on charts.
import { METHOD_META } from '../constants';

export default function MethodToggle({ available, enabled, setEnabled }) {
  function toggle(m) {
    setEnabled(prev =>
      prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
    );
  }

  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {available.map(m => {
        const meta = METHOD_META[m] || { label: m, color: '#aaa', isNovel: false };
        const on   = enabled.includes(m);
        return (
          <button
            key={m}
            onClick={() => toggle(m)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
                        border transition-all duration-150 ${
              on
                ? 'border-transparent text-white'
                : 'bg-transparent border-white/15 text-white/40 hover:text-white/60'
            }`}
            style={on ? { backgroundColor: meta.color + '33', borderColor: meta.color + '88' } : {}}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: on ? meta.color : '#555' }}
            />
            {meta.label}
            {meta.isNovel && (
              <span className="ml-1 text-[9px] bg-pink-500/40 text-pink-300 px-1 rounded">NEW</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
