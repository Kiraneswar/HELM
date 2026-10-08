// src/App.jsx
import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout     from './components/Layout';
import { useResults } from './hooks/useResults';

import Overview        from './pages/Overview';
import AccuracyCurves  from './pages/AccuracyCurves';
import FairnessLoss    from './pages/FairnessLoss';
import EnergyCost      from './pages/EnergyCost';
import Tradeoff        from './pages/Tradeoff';
import RLDiagnostics   from './pages/RLDiagnostics';
import ComparisonTable from './pages/ComparisonTable';

function LoadingScreen() {
  return (
    <div className="flex items-center justify-center h-full min-h-[60vh]">
      <div className="text-center">
        <div className="w-12 h-12 border-2 border-violet-500 border-t-transparent
                        rounded-full animate-spin mx-auto mb-4" />
        <div className="text-white/50 text-sm">Loading experiment results…</div>
      </div>
    </div>
  );
}

function ErrorScreen({ message, onRetry }) {
  return (
    <div className="flex items-center justify-center h-full min-h-[60vh]">
      <div className="glass rounded-xl p-8 max-w-md text-center">
        <div className="text-red-400 text-4xl mb-4">⚠</div>
        <h3 className="text-white font-semibold mb-2">Failed to load data</h3>
        <p className="text-white/40 text-sm mb-4 font-mono">{message}</p>
        <p className="text-white/30 text-xs mb-6">
          Make sure the FastAPI backend is running:<br/>
          <code className="text-violet-400">uvicorn backend.main:app --reload --port 8000</code>
        </p>
        <button
          onClick={onRetry}
          className="bg-violet-600 hover:bg-violet-500 text-white text-sm px-4 py-2 rounded-lg">
          Retry
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [resultsDir, setResultsDir] = useState('');
  const { summary, allResults, available, loading, error, reload } = useResults(resultsDir || undefined);

  function renderContent() {
    if (loading)       return <LoadingScreen />;
    if (error)         return <ErrorScreen message={error} onRetry={reload} />;
    return (
      <Routes>
        <Route path="/"         element={<Overview        summary={summary} />} />
        <Route path="/accuracy" element={<AccuracyCurves  allResults={allResults} available={available} />} />
        <Route path="/fairness" element={<FairnessLoss    allResults={allResults} available={available} />} />
        <Route path="/energy"   element={<EnergyCost      allResults={allResults} available={available} />} />
        <Route path="/tradeoff" element={<Tradeoff        summary={summary} />} />
        <Route path="/rl"       element={<RLDiagnostics   allResults={allResults} />} />
        <Route path="/table"    element={<ComparisonTable summary={summary} allResults={allResults} />} />
      </Routes>
    );
  }

  return (
    <BrowserRouter>
      <Layout
        resultsDir={resultsDir}
        setResultsDir={setResultsDir}
        onReload={reload}
        loading={loading}
      >
        {renderContent()}
      </Layout>
    </BrowserRouter>
  );
}
