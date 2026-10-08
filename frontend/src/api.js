// src/api.js
// ----------
// Thin wrapper around the FastAPI backend endpoints.
// All React components import from here — never fetch directly.

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function resultsParam(resultsDir) {
  return resultsDir ? `?results_dir=${encodeURIComponent(resultsDir)}` : '';
}

export async function fetchMethods(resultsDir) {
  const res = await fetch(`${BASE}/api/methods${resultsParam(resultsDir)}`);
  if (!res.ok) throw new Error(`/api/methods failed: ${res.status}`);
  return res.json();
}

export async function fetchResults(method, resultsDir) {
  const res = await fetch(`${BASE}/api/results/${method}${resultsParam(resultsDir)}`);
  if (!res.ok) throw new Error(`/api/results/${method} failed: ${res.status}`);
  return res.json();
}

export async function fetchAllResults(methods, resultsDir) {
  const settled = await Promise.allSettled(
    methods.map(m => fetchResults(m, resultsDir))
  );
  return settled
    .filter(r => r.status === 'fulfilled')
    .map(r => r.value);
}

export async function fetchSummary(resultsDir) {
  const res = await fetch(`${BASE}/api/summary${resultsParam(resultsDir)}`);
  if (!res.ok) throw new Error(`/api/summary failed: ${res.status}`);
  return res.json();
}

export async function fetchConfig(resultsDir) {
  const res = await fetch(`${BASE}/api/config${resultsParam(resultsDir)}`);
  if (!res.ok) throw new Error(`/api/config failed: ${res.status}`);
  return res.json();
}
