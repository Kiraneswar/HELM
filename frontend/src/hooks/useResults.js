// src/hooks/useResults.js
// Fetches and caches all method results from the backend.
import { useState, useEffect, useCallback } from 'react';
import { fetchSummary, fetchAllResults, fetchMethods } from '../api';
import { ALL_METHODS } from '../constants';

export function useResults(resultsDir) {
  const [summary, setSummary]         = useState(null);
  const [allResults, setAllResults]   = useState(null);
  const [available, setAvailable]     = useState([]);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumData, methData] = await Promise.all([
        fetchSummary(resultsDir),
        fetchMethods(resultsDir),
      ]);
      const avail = methData.methods.map(m => m.method);
      setAvailable(avail);
      setSummary(sumData.summary);

      const results = await fetchAllResults(avail, resultsDir);
      setAllResults(results);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [resultsDir]);

  useEffect(() => { load(); }, [load]);

  return { summary, allResults, available, loading, error, reload: load };
}
