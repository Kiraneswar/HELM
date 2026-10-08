// src/components/ChartWrapper.jsx
// Consistent card wrapper for every Recharts chart.
export default function ChartWrapper({ title, subtitle, children, className = '' }) {
  return (
    <div className={`glass rounded-xl p-5 ${className}`}>
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white/90">{title}</h3>
        {subtitle && <p className="text-xs text-white/40 mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}
