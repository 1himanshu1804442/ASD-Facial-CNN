import React from 'react';
import Header from '../components/Header';
import ModelBenchmarksView from '../components/ModelBenchmarksView';
import { useNeuroGraph } from '../hooks/useNeuroGraph';

/**
 * BenchmarkComparison Page.
 * 
 * Why: Dedicated clinical comparison view for comparing deep learning vision architectures
 * (Mod-Stem ResNet-50 + LSTM vs standard ResNet-50, DenseNet-121, EfficientNet-B0, MobileNet-V2, ViT-B/16).
 * 
 * Provides:
 * - State management and background health polling via useNeuroGraph.
 * - Strict null-safe data lookups (e.g., benchmarkData?.modstem_lstm, benchmarkData?.resnet50).
 * - Division-by-zero protection via ModelMetricsTable.
 * - Seamless React Bounce compliant dark background base.
 */
export default function BenchmarkComparison({
  onNavigateToScreening,
  benchmarkData,
}) {
  const {
    isOnline,
    patientHistory,
    isHistoryModalOpen,
    setIsHistoryModalOpen,
    handleSelectHistoricalPatient,
  } = useNeuroGraph();

  return (
    <div
      className="flex flex-col min-h-screen bg-[#080c14] text-slate-100"
      style={{
        margin: 0,
        padding: 0,
        width: '100%',
        minHeight: '100vh',
        backgroundColor: '#080c14',
      }}
    >
      <Header
        isOnline={isOnline}
        onOpenHistory={() => setIsHistoryModalOpen(true)}
        historyCount={patientHistory?.length || 0}
      />

      <main className="max-w-[1440px] w-full mx-auto p-5 md:p-7 flex flex-col gap-6 flex-1">
        <ModelBenchmarksView
          onSwitchToScreening={onNavigateToScreening}
          benchmarkData={benchmarkData}
        />
      </main>
    </div>
  );
}
