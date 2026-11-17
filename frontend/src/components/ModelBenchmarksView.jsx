import React from 'react';
import {
  Award,
  CheckCircle,
  ExternalLink,
  Cpu,
  BarChart3,
  Database,
  ShieldAlert,
  Sparkles,
  ArrowLeft,
  Scan,
} from 'lucide-react';
import ModelMetricsTable, { DEFAULT_BENCHMARK_DATA } from './ModelMetricsTable';

export default function ModelBenchmarksView({ onSwitchToScreening, benchmarkData = DEFAULT_BENCHMARK_DATA }) {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-cyan-950/40 via-blue-950/30 to-indigo-950/40 border border-cyan-500/30 rounded-2xl p-6 relative overflow-hidden backdrop-blur-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-semibold tracking-wide uppercase">
                <Sparkles className="w-3.5 h-3.5" />
                Empirical Literature Comparison
              </div>

              {onSwitchToScreening && (
                <button
                  type="button"
                  onClick={onSwitchToScreening}
                  className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-500/15 border border-indigo-500/35 text-indigo-300 text-xs font-semibold hover:bg-indigo-500/25 transition-colors cursor-pointer"
                >
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Return to Facial Screening</span>
                </button>
              )}
            </div>

            <h2 className="text-2xl font-bold text-white tracking-tight">
              Model Performance & Benchmark Analysis
            </h2>
            <p className="text-sm text-slate-300 max-w-3xl">
              Quantitative comparison of our proposed <strong>Facial Landmark GCN</strong> against baseline Convolutional Neural Networks (Ahmad et al., 2024) and Vision Transformers (Contreras et al., 2025) on the standardized Zenodo ASD dataset.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="bg-slate-900/80 border border-slate-700/60 rounded-xl px-4 py-2.5 text-center">
              <span className="block text-xs font-medium text-slate-400 uppercase tracking-wider">MobileNet-V2</span>
              <span className="text-xl font-bold text-cyan-400">2.2M</span>
              <span className="block text-[10px] text-emerald-400 font-medium">24 ms Edge Latency</span>
            </div>
            <div className="bg-slate-900/80 border border-slate-700/60 rounded-xl px-4 py-2.5 text-center">
              <span className="block text-xs font-medium text-slate-400 uppercase tracking-wider">Top AUROC</span>
              <span className="text-xl font-bold text-emerald-400">0.944</span>
              <span className="block text-[10px] text-emerald-400 font-medium">DenseNet-121</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Sortable & Division-by-Zero Protected Benchmark Table */}
      <ModelMetricsTable
        benchmarkData={benchmarkData}
        onSelectArchitecture={() => {
          if (onSwitchToScreening) onSwitchToScreening();
        }}
      />

      {/* Key Academic Citations & References */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-cyan-400">
            <Database className="w-4 h-4" />
            <h4 className="font-semibold text-slate-200 text-sm">Zenodo Benchmark Repository</h4>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            "ASD Detection from Facial Images – Dataset, Code and Results" by Rodrigo Colnago Contreras (Universidade Federal de São Paulo, 2025).
          </p>
          <a
            href="https://zenodo.org/records/15073612"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            DOI: 10.5281/zenodo.15073612 <ExternalLink className="w-3 h-3" />
          </a>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-indigo-400">
            <Cpu className="w-4 h-4" />
            <h4 className="font-semibold text-slate-200 text-sm">Baseline CNN Study</h4>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            "Autism spectrum disorder detection using facial images: A performance comparison of pretrained CNNs" by Ahmad et al., <em>Healthcare Technology Letters</em> (2024).
          </p>
          <span className="text-xs text-indigo-400 font-mono">
            PubMed Central: PMC11340156
          </span>
        </div>

        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400">
            <Award className="w-4 h-4" />
            <h4 className="font-semibold text-slate-200 text-sm">Methodological Standard</h4>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Modeled after Singh, Shukla, & Gore, <em>Arabian Journal for Science and Engineering</em> (Springer, 2024) from MNNIT Allahabad.
          </p>
          <span className="text-xs text-emerald-400 font-mono">
            DOI: 10.1007/s13369-024-09362-2
          </span>
        </div>
      </div>
    </div>
  );
}
