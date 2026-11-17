import React from 'react';
import Header from '../components/Header';
import HistoricalRegistryView from '../components/HistoricalRegistryView';
import PatientHistoryModal from '../components/PatientHistoryModal';
import { useNeuroGraph } from '../hooks/useNeuroGraph';
import { AlertCircle } from 'lucide-react';

/**
 * PatientReports Page.
 * 
 * Why: Dedicated clinical report and historical registry view for reviewing
 * persisted PostgreSQL screening records, with resilient error states if
 * Spring Boot is slow or returns 500.
 */
export default function PatientReports({ onNavigateToScreening }) {
  const {
    isOnline,
    patientHistory,
    isHistoryLoading,
    isHistoryError,
    historyErrorMessage,
    refetchHistory,
    handleClearHistory,
    isHistoryClearing,
    handleSeedAbide,
    isSeedingAbide,
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
        {/* Error notification if Spring Boot returns 500 or is unreachable */}
        {isHistoryError && historyErrorMessage && (
          <div className="bg-red-500/15 border border-red-500/35 rounded-xl p-3.5 flex items-center gap-3 text-red-300 text-sm">
            <AlertCircle size={18} className="shrink-0 text-red-400" />
            <span>
              Backend registry warning: {historyErrorMessage}. Fallback clinical preview records active.
            </span>
          </div>
        )}

        <HistoricalRegistryView
          records={patientHistory}
          isLoading={isHistoryLoading}
          error={historyErrorMessage}
          onRefresh={refetchHistory}
          onClearHistory={handleClearHistory}
          isClearing={isHistoryClearing}
          onSeedAbide={handleSeedAbide}
          isSeeding={isSeedingAbide}
          onSelectPatient={handleSelectHistoricalPatient}
          onSwitchToScreening={onNavigateToScreening}
        />
      </main>

      <PatientHistoryModal
        isOpen={isHistoryModalOpen}
        onClose={() => setIsHistoryModalOpen(false)}
        records={patientHistory}
        isLoading={isHistoryLoading}
        error={historyErrorMessage}
        onRefresh={refetchHistory}
        onClearHistory={handleClearHistory}
        isClearing={isHistoryClearing}
        onSeedAbide={handleSeedAbide}
        isSeeding={isSeedingAbide}
        onSelectPatient={(patient) => {
          handleSelectHistoricalPatient(patient);
          setIsHistoryModalOpen(false);
          if (onNavigateToScreening) onNavigateToScreening();
        }}
      />
    </div>
  );
}
