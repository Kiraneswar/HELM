// src/constants.js
// Method metadata — single source of truth, mirrors backend METHOD_COLORS/LABELS

export const METHOD_META = {
  random:         { label: 'Random',                 color: '#9E9E9E', isNovel: false },
  full:           { label: 'Full Participation',     color: '#4CAF50', isNovel: false },
  oort:           { label: 'Oort (Heuristic)',        color: '#FF9800', isNovel: false },
  centralized_rl: { label: 'Centralized RL',          color: '#2196F3', isNovel: false },
  federated_marl: { label: 'Federated MARL (Ours)',   color: '#E91E63', isNovel: true  },
};

export const ALL_METHODS = Object.keys(METHOD_META);

export const NOVEL_METHOD = 'federated_marl';

export const CHART_THEME = {
  gridColor:   'rgba(255,255,255,0.06)',
  axisColor:   'rgba(255,255,255,0.35)',
  tooltipBg:   'rgba(6,0,60,0.95)',
  tooltipBorder: 'rgba(139,92,246,0.5)',
};
