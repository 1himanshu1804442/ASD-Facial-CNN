import React, { useState, useMemo } from 'react';
import {
  BarChart3,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Award,
  CheckCircle2,
  ExternalLink,
  Cpu,
  Zap,
} from 'lucide-react';

/**
 * Safe division helper preventing division-by-zero, NaN, and Infinity runtime crashes.
 * Why: If sample count is 0 or confusion matrix sums are empty, standard division yields
 * NaN or Infinity, corrupting table sorting and rendering.
 * 
 * @param {number} num - Numerator
 * @param {number} den - Denominator
 * @param {number} [fallback=0.0] - Safe return value if denominator <= 0 or non-finite
 * @returns {number} Safe quotient
 */
export function safeDivide(num, den, fallback = 0.0) {
  if (typeof num !== 'number' || typeof den !== 'number' || isNaN(num) || isNaN(den)) {
    return fallback;
  }
  if (den <= 0 || !isFinite(den)) {
    return fallback;
  }
  const result = num / den;
  return isFinite(result) ? result : fallback;
}

/**
 * Safe metric conversion helper converting raw decimal scores to formatted percentage strings.
 * 
 * @param {number|string} val - Decimal or percentage number (e.g. 0.8533 or '85.33%')
 * @returns {string} Formatted percentage (e.g. '85.33%')
 */
export function formatMetricPercentage(val) {
  if (val == null) return '0.00%';
  if (typeof val === 'string' && val.includes('%')) return val;
  const num = Number(val);
  if (isNaN(num)) return '0.00%';
  const pct = num <= 1.0 ? num * 100 : num;
  return `${pct.toFixed(2)}%`;
}

/**
 * Safe metric conversion helper converting latency values to formatted millisecond strings.
 * 
 * @param {number|string} val - Latency in ms (e.g. 91.2 or '91 ms')
 * @returns {string} Formatted latency (e.g. '91.2 ms')
 */
export function formatLatency(val) {
  if (val == null) return '0.0 ms';
  if (typeof val === 'string' && val.includes('ms')) return val;
  const num = Number(val);
  if (isNaN(num)) return '0.0 ms';
  return `${num.toFixed(1)} ms`;
}

// Canonical literature benchmark cohort dataset (Zenodo 15073612 held-out 300-patient cohort)
export const DEFAULT_BENCHMARK_DATA = {
  modstem_lstm: {
    key: 'modstem_lstm',
    architecture: 'Mod-Stem ResNet-50 + Spatial Bi-LSTM (Proposed)',
    type: 'Mod-Stem 3x3 Conv + 7-Tier Spatial Bi-LSTM',
    literature_citation: 'Proposed Architecture (This Work, 2026)',
    parameters_m: 25.8,
    accuracy: 0.84,
    sensitivity: 0.88,
    specificity: 0.8,
    precision: 0.8148,
    f1_score: 0.8462,
    auc_roc: 0.9139,
    inference_latency_ms: 0.28,
    true_positives: 132,
    true_negatives: 120,
    false_positives: 30,
    false_negatives: 18,
    samples_evaluated: 300,
    xai: 'Topological 68-Landmark Saliency & Heatmaps',
    highlight: true,
  },
  resnet50: {
    key: 'resnet50',
    architecture: 'ResNet-50 (Residual Baseline)',
    type: 'Residual 2D Bottleneck CNN',
    literature_citation: 'Khan et al. [11] / Monani et al. [13]',
    parameters_m: 23.5,
    accuracy: 0.86,
    sensitivity: 0.8067,
    specificity: 0.9133,
    precision: 0.903,
    f1_score: 0.8521,
    auc_roc: 0.9514,
    inference_latency_ms: 0.32,
    true_positives: 121,
    true_negatives: 137,
    false_positives: 13,
    false_negatives: 29,
    samples_evaluated: 300,
    xai: 'Grad-CAM only (Layer4)',
    highlight: false,
  },
  mobilenet_v2: {
    key: 'mobilenet_v2',
    architecture: 'MobileNet-V2 (Lightweight Fast Triage)',
    type: 'Inverted Residual Mobile CNN',
    literature_citation: 'Beary et al. [1] / Alkahtani et al. [7]',
    parameters_m: 2.2,
    accuracy: 0.8867,
    sensitivity: 0.9067,
    specificity: 0.8667,
    precision: 0.8718,
    f1_score: 0.8889,
    auc_roc: 0.9448,
    inference_latency_ms: 0.23,
    true_positives: 136,
    true_negatives: 130,
    false_positives: 20,
    false_negatives: 14,
    samples_evaluated: 300,
    xai: 'Grad-CAM only (Conv2d)',
    highlight: false,
  },
  densenet121: {
    key: 'densenet121',
    architecture: 'DenseNet-121 (Dense Feature-Reuse)',
    type: 'Dense Feature Reuse CNN',
    literature_citation: 'Khan et al. [11] (96.0% Acc on ASD)',
    parameters_m: 7.0,
    accuracy: 0.85,
    sensitivity: 0.8533,
    specificity: 0.8467,
    precision: 0.8477,
    f1_score: 0.8505,
    auc_roc: 0.9296,
    inference_latency_ms: 0.62,
    true_positives: 128,
    true_negatives: 127,
    false_positives: 23,
    false_negatives: 22,
    samples_evaluated: 300,
    xai: 'Grad-CAM only (DenseBlock4)',
    highlight: false,
  },
  efficientnet_b0: {
    key: 'efficientnet_b0',
    architecture: 'EfficientNet-B0 (Compound Scaled)',
    type: 'Compound Scaled CNN',
    literature_citation: 'Rahman & Subashini [2] (96.6% AUC on Facial Screening)',
    parameters_m: 4.0,
    accuracy: 0.8367,
    sensitivity: 0.86,
    specificity: 0.8133,
    precision: 0.8217,
    f1_score: 0.8404,
    auc_roc: 0.9142,
    inference_latency_ms: 0.35,
    true_positives: 129,
    true_negatives: 122,
    false_positives: 28,
    false_negatives: 21,
    samples_evaluated: 300,
    xai: 'Grad-CAM only (MBConv)',
    highlight: false,
  },
  vit: {
    key: 'vit',
    architecture: 'ViT-B/16 (Vision Transformer)',
    type: 'Pure Vision Transformer (Patch 16)',
    literature_citation: 'Revathi & Arunarani [3] (ViT Attention on ASD)',
    parameters_m: 85.8,
    accuracy: 0.87,
    sensitivity: 0.8733,
    specificity: 0.8667,
    precision: 0.8675,
    f1_score: 0.8704,
    auc_roc: 0.9452,
    inference_latency_ms: 0.22,
    true_positives: 131,
    true_negatives: 130,
    false_positives: 20,
    false_negatives: 19,
    samples_evaluated: 300,
    xai: 'Attention Rollout (Patch-level)',
    highlight: false,
  },
};

/**
 * ModelMetricsTable Component.
 * 
 * Why: Renders an interactive, sortable benchmark performance matrix comparing
 * all evaluated vision backbones. Implements strict division-by-zero protection,
 * null-safe data lookups for benchmarkData entries, and metric conversions.
 * 
 * @param {Object} props
 * @param {Object|Array} [props.benchmarkData] - Custom or backend-provided benchmark dictionary.
 * @param {Function} [props.onSelectArchitecture] - Optional callback when clinician picks an architecture.
 */
export default function ModelMetricsTable({
  benchmarkData = DEFAULT_BENCHMARK_DATA,
  onSelectArchitecture,
}) {
  // Sort configuration state
  const [sortKey, setSortKey] = useState('accuracy');
  const [sortOrder, setSortOrder] = useState('desc'); // 'asc' | 'desc'

  /**
   * Normalizes benchmark data from object dictionary or array with null-safe lookups.
   * Why: Supports both raw JSON object mapping { modstem_lstm: {...}, ... } and direct arrays.
   */
  const normalizedRows = useMemo(() => {
    let sourceList = [];

    if (Array.isArray(benchmarkData)) {
      sourceList = benchmarkData;
    } else if (benchmarkData && typeof benchmarkData === 'object') {
      // Check individual known architectures with null-safe lookups
      const keys = ['modstem_lstm', 'resnet50', 'mobilenet_v2', 'densenet121', 'efficientnet_b0', 'vit'];
      sourceList = keys
        .map((k) => benchmarkData?.[k] || DEFAULT_BENCHMARK_DATA[k])
        .filter(Boolean);

      // Append any other dynamically keyed entries in benchmarkData
      Object.keys(benchmarkData).forEach((k) => {
        if (!keys.includes(k) && benchmarkData[k] && typeof benchmarkData[k] === 'object') {
          sourceList.push(benchmarkData[k]);
        }
      });
    } else {
      sourceList = Object.values(DEFAULT_BENCHMARK_DATA);
    }

    // Map each entry through division-by-zero protection
    return sourceList.map((row, idx) => {
      const tp = Number(row?.true_positives ?? 0);
      const tn = Number(row?.true_negatives ?? 0);
      const fp = Number(row?.false_positives ?? 0);
      const fn = Number(row?.false_negatives ?? 0);
      const samples = Number(row?.samples_evaluated ?? tp + tn + fp + fn);

      // Division-by-zero guarded calculations
      const computedAccuracy = samples > 0 ? safeDivide(tp + tn, samples) : 0.0;
      const computedSensitivity = (tp + fn) > 0 ? safeDivide(tp, tp + fn) : 0.0;
      const computedSpecificity = (tn + fp) > 0 ? safeDivide(tn, tn + fp) : 0.0;
      const computedPrecision = (tp + fp) > 0 ? safeDivide(tp, tp + fp) : 0.0;

      const accuracyVal = typeof row?.accuracy === 'number' ? row.accuracy : computedAccuracy;
      const sensitivityVal = typeof row?.sensitivity === 'number' ? row.sensitivity : computedSensitivity;
      const specificityVal = typeof row?.specificity === 'number' ? row.specificity : computedSpecificity;
      const precisionVal = typeof row?.precision === 'number' ? row.precision : computedPrecision;
      const f1Val = typeof row?.f1_score === 'number'
        ? row.f1_score
        : (typeof row?.f1 === 'number' ? row.f1 : safeDivide(2 * precisionVal * sensitivityVal, precisionVal + sensitivityVal));
      const aurocVal = typeof row?.auc_roc === 'number' ? row.auc_roc : (typeof row?.auroc === 'number' ? row.auroc : 0.90);
      const latencyVal = typeof row?.inference_latency_ms === 'number' ? row.inference_latency_ms : (typeof row?.latency === 'number' ? row.latency : 50.0);
      const paramsVal = typeof row?.parameters_m === 'number' ? row.parameters_m : (typeof row?.params === 'number' ? row.params : 25.0);

      return {
        id: row?.key || row?.architecture || `model-${idx}`,
        architecture: row?.architecture || row?.model || 'Unknown Architecture',
        type: row?.type || 'Neural Backbone',
        literature_citation: row?.literature_citation || row?.citation || 'Literature Standard',
        parameters_m: paramsVal,
        accuracy: accuracyVal,
        sensitivity: sensitivityVal,
        specificity: specificityVal,
        precision: precisionVal,
        f1_score: f1Val,
        auc_roc: aurocVal,
        inference_latency_ms: latencyVal,
        samples_evaluated: samples,
        xai: row?.xai || 'Explainability Map',
        highlight: Boolean(row?.highlight),
      };
    });
  }, [benchmarkData]);

  /**
   * Sorts normalized rows based on active column and direction.
   */
  const sortedRows = useMemo(() => {
    return [...normalizedRows].sort((a, b) => {
      let valA = a[sortKey];
      let valB = b[sortKey];

      if (typeof valA === 'string') {
        valA = valA.toLowerCase();
        valB = String(valB || '').toLowerCase();
        return sortOrder === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }

      const numA = Number(valA ?? 0);
      const numB = Number(valB ?? 0);
      return sortOrder === 'asc' ? numA - numB : numB - numA;
    });
  }, [normalizedRows, sortKey, sortOrder]);

  /**
   * Handles column header click to toggle sorting direction or set new sort key.
   */
  const handleSort = (key) => {
    if (sortKey === key) {
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortOrder('desc'); // Default to descending for clinical performance metrics
    }
  };

  /**
   * Renders sort direction indicator icon.
   */
  const renderSortIndicator = (key) => {
    if (sortKey !== key) {
      return <ArrowUpDown size={12} className="text-slate-500 opacity-60 group-hover:opacity-100" />;
    }
    return sortOrder === 'asc' ? (
      <ArrowUp size={12} className="text-cyan-400" />
    ) : (
      <ArrowDown size={12} className="text-cyan-400" />
    );
  };

  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl backdrop-blur-sm">
      {/* Table Subtitle Bar */}
      <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40 flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <BarChart3 className="w-5 h-5 text-cyan-400" />
          <h3 className="font-semibold text-slate-200">
            Table 1: Performance Matrix Across Vision Architectures
          </h3>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-400 font-mono">
            Held-Out Test Cohort (N = 300) • Sortable
          </span>
        </div>
      </div>

      {/* Responsive Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-800/80 bg-slate-950/60 text-xs text-slate-400 uppercase tracking-wider font-semibold">
              {/* Architecture Column */}
              <th
                onClick={() => handleSort('architecture')}
                className="py-3.5 px-6 cursor-pointer hover:text-white transition-colors group select-none"
              >
                <div className="flex items-center gap-1.5">
                  <span>Model &amp; Study</span>
                  {renderSortIndicator('architecture')}
                </div>
              </th>

              {/* Parameters Column */}
              <th
                onClick={() => handleSort('parameters_m')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Params (M)</span>
                  {renderSortIndicator('parameters_m')}
                </div>
              </th>

              {/* Accuracy Column */}
              <th
                onClick={() => handleSort('accuracy')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Accuracy</span>
                  {renderSortIndicator('accuracy')}
                </div>
              </th>

              {/* Sensitivity Column */}
              <th
                onClick={() => handleSort('sensitivity')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Sensitivity</span>
                  {renderSortIndicator('sensitivity')}
                </div>
              </th>

              {/* Specificity Column */}
              <th
                onClick={() => handleSort('specificity')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Specificity</span>
                  {renderSortIndicator('specificity')}
                </div>
              </th>

              {/* F1-Score Column */}
              <th
                onClick={() => handleSort('f1_score')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>F1-Score</span>
                  {renderSortIndicator('f1_score')}
                </div>
              </th>

              {/* AUROC Column */}
              <th
                onClick={() => handleSort('auc_roc')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>AUROC</span>
                  {renderSortIndicator('auc_roc')}
                </div>
              </th>

              {/* Latency Column */}
              <th
                onClick={() => handleSort('inference_latency_ms')}
                className="py-3.5 px-4 cursor-pointer hover:text-white transition-colors group select-none text-center"
                title="Batch inference latency measured on Nvidia Tesla T4 GPU in Colab runtime"
              >
                <div className="flex items-center justify-center gap-1.5">
                  <span>Latency (T4 GPU)</span>
                  {renderSortIndicator('inference_latency_ms')}
                </div>
              </th>

              {/* Interpretability / XAI */}
              <th className="py-3.5 px-6">Clinical Interpretability (XAI)</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
            {sortedRows.map((row) => {
              const isSelected = Boolean(row.highlight);

              return (
                <tr
                  key={row.id}
                  onClick={() => onSelectArchitecture && onSelectArchitecture(row.architecture)}
                  className={`transition-colors ${
                    isSelected
                      ? 'bg-cyan-950/30 hover:bg-cyan-950/40 border-l-4 border-l-cyan-500 font-semibold'
                      : 'hover:bg-slate-800/30'
                  } ${onSelectArchitecture ? 'cursor-pointer' : ''}`}
                >
                  {/* Model & Study Description */}
                  <td className="py-4 px-6 font-sans text-sm font-medium text-slate-200">
                    <div className="flex items-center gap-2">
                      {isSelected && <Award className="w-4 h-4 text-amber-400 shrink-0" />}
                      <div className="flex flex-col">
                        <span className="font-bold text-slate-100">{row.architecture}</span>
                        <span className="text-xs text-slate-400">{row.literature_citation}</span>
                      </div>
                    </div>
                  </td>

                  {/* Parameter Count */}
                  <td className="py-4 px-4 text-center text-slate-400">
                    {row.parameters_m.toFixed(1)}M
                  </td>

                  {/* Accuracy Badge */}
                  <td className="py-4 px-4 text-center text-slate-200 font-bold">
                    <span
                      className={`px-2.5 py-1 rounded-full ${
                        isSelected
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                          : 'bg-slate-800 text-slate-300'
                      }`}
                    >
                      {formatMetricPercentage(row.accuracy)}
                    </span>
                  </td>

                  {/* Sensitivity */}
                  <td className="py-4 px-4 text-center text-slate-300">
                    {formatMetricPercentage(row.sensitivity)}
                  </td>

                  {/* Specificity */}
                  <td className="py-4 px-4 text-center text-slate-300">
                    {formatMetricPercentage(row.specificity)}
                  </td>

                  {/* F1-Score */}
                  <td className="py-4 px-4 text-center text-slate-300 font-semibold">
                    {typeof row.f1_score === 'number' ? row.f1_score.toFixed(3) : row.f1_score}
                  </td>

                  {/* AUROC */}
                  <td className="py-4 px-4 text-center text-emerald-400 font-bold">
                    {typeof row.auc_roc === 'number' ? row.auc_roc.toFixed(3) : row.auc_roc}
                  </td>

                  {/* Real-time Edge Latency */}
                  <td className="py-4 px-4 text-center text-slate-300">
                    <span className={`inline-flex items-center gap-1 ${row.inference_latency_ms < 50 ? 'text-emerald-400 font-bold' : 'text-slate-300'}`}>
                      {row.inference_latency_ms < 50 && <Zap className="w-3 h-3" />}
                      {formatLatency(row.inference_latency_ms)}
                    </span>
                  </td>

                  {/* Clinical Interpretability */}
                  <td className="py-4 px-6 font-sans text-xs text-slate-300">
                    <span
                      className={`inline-flex items-center gap-1.5 ${
                        isSelected ? 'text-emerald-300 font-medium' : 'text-slate-400'
                      }`}
                    >
                      {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                      {row.xai}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Benchmark Footnote & Context Metadata */}
      <div className="px-6 py-3.5 bg-slate-950/70 border-t border-slate-800/80 text-xs text-slate-400 flex flex-col md:flex-row md:items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-slate-300">Dataset Citation:</span>
          <span>ASD Detection from Facial Images – Dataset, Code and Results (Contreras, UNIFESP, 2025; Zenodo DOI: 10.5281/zenodo.15073612)</span>
        </div>
        <div className="text-[11px] text-slate-400">
          * Latency indicates provisional batched T4 GPU timing (0.22–0.62 ms/image). Edge CPU deployment latency: ~24–85 ms. Preliminary research screening aid only.
        </div>
      </div>
    </div>
  );
}
