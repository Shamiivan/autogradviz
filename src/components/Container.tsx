import { useState, useEffect, useMemo, type CSSProperties } from "react";
import { TrainingManager } from "../micrograd/training";
import type { TrainingSnapshot } from "../micrograd/types";
import { NetworkVisualizer } from "./NetworkVizualiser";
import { FeatureScatter } from "./FeatureScatter";
import { LegendOverlay } from "./LegendOverlay";
import { IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES } from "../utils";
import { MetricsChart, type MetricsPoint } from "./MetricsChart";

function Container() {
  const [trainingManager, setTrainingManager] = useState<TrainingManager | null>(null);
  const [snapshots, setSnapshots] = useState<TrainingSnapshot[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationTrigger, setAnimationTrigger] = useState(0);
  const [maxEpochs, setMaxEpochs] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [viewportWidth, setViewportWidth] = useState(
    typeof window !== "undefined" ? window.innerWidth : 1024
  );
  const isMobile = viewportWidth < 768;
  const ANIMATION_PAUSE_MESSAGE = "Training paused for visualization – press Resume to continue";
  const MANUAL_PAUSE_MESSAGE = "Training paused – press Resume to continue";

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handleResize = () => setViewportWidth(window.innerWidth);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Initialize training manager and load data
  useEffect(() => {
    const initTraining = async () => {
      console.log("Initializing training system...");

      // Create training manager with config - more epochs for better visualization
      const manager = new TrainingManager({
        epochs: 10,
        learningRate: 0.1,
        trainTestSplit: 0.8,
        stepDelayMs: 120
      });

      // Load Iris data
      await manager.loadData();

      setTrainingManager(manager);
      setMaxEpochs(manager.getTotalEpochs());
      setIsLoading(false);

      console.log("System ready! Click 'Start Training' to begin.");
    };

    initTraining();
  }, []);

  // Start training
  const handleStartTraining = async () => {
    if (!trainingManager || isTraining) return;

    setStatusMessage(null);
    setSnapshots([]);
    setCurrentEpoch(0);
    setIsPaused(false);
    trainingManager.clearSnapshots();

    const unsubscribe = trainingManager.onSnapshot(snapshot => {
      setSnapshots(prev => [...prev, snapshot]);
      setCurrentEpoch(snapshot.epoch);
    });

    setIsTraining(true);
    console.log("\n" + "=".repeat(50));

    try {
      // Train the network
      await trainingManager.train();

      const allSnapshots = trainingManager.getSnapshots();

      console.log("=".repeat(50) + "\n");

      // Log summary
      console.log("Training Summary:");
      console.log(`   Total steps: ${allSnapshots.length}`);
      console.log(`   Initial loss: ${allSnapshots[0]?.loss.toFixed(4)}`);
      console.log(`   Final loss: ${allSnapshots[allSnapshots.length - 1]?.loss.toFixed(4)}`);
    } finally {
      unsubscribe();
      setIsTraining(false);
      setIsPaused(false);
      setStatusMessage(null);
    }
  };

  const handlePauseTraining = () => {
    if (!trainingManager || !isTraining || isPaused) return;
    trainingManager.pause();
    setIsPaused(true);
    setStatusMessage(MANUAL_PAUSE_MESSAGE);
  };

  const handleResumeTraining = () => {
    if (!trainingManager || !isTraining || !isPaused) return;
    trainingManager.resume();
    setIsPaused(false);
    setStatusMessage(null);
  };

  // Play animation for current epoch
  const handlePlayAnimation = () => {
    if (snapshots.length === 0 || isAnimating) return;

    if (trainingManager && isTraining && !isPaused) {
      trainingManager.pause();
      setIsPaused(true);
      setStatusMessage(ANIMATION_PAUSE_MESSAGE);
    }

    setIsAnimating(true);
    setAnimationTrigger(prev => prev + 1);
  };

  // Go to next epoch
  const handleNextEpoch = () => {
    if (currentEpoch < maxEpochs - 1) {
      setCurrentEpoch(prev => prev + 1);
    }
  };

  // Go to previous epoch
  const handlePrevEpoch = () => {
    if (currentEpoch > 0) {
      setCurrentEpoch(prev => prev - 1);
    }
  };

  // Animation complete callback
  const handleAnimationComplete = () => {
    setIsAnimating(false);
  };

  // Get snapshots for current epoch
  const getCurrentEpochSnapshots = () => {
    return snapshots.filter(s => s.epoch === currentEpoch);
  };

  // Get a representative snapshot from current epoch (last one)
  const getCurrentSnapshot = () => {
    const epochSnapshots = getCurrentEpochSnapshots();
    return epochSnapshots.length > 0 ? epochSnapshots[epochSnapshots.length - 1] : undefined;
  };

  const currentEpochSnapshots = getCurrentEpochSnapshots();
  const currentSnapshot = getCurrentSnapshot();

  const currentEpochAvgLoss = currentEpochSnapshots.length > 0
    ? currentEpochSnapshots.reduce((sum, s) => sum + s.loss, 0) / currentEpochSnapshots.length
    : 0;

  const currentEpochAccuracy = currentEpochSnapshots.length > 0
    ? (currentEpochSnapshots.filter(s => s.prediction === s.actualClass).length / currentEpochSnapshots.length) * 100
    : 0;

  const labeledInputs = currentSnapshot
    ? (currentSnapshot.activations[0] || []).map((value, idx) => ({
      label: IRIS_FEATURE_NAMES[idx] ?? `Feature ${idx + 1}`,
      value
    }))
    : [];

  const outputActivations = currentSnapshot
    ? currentSnapshot.activations[currentSnapshot.activations.length - 1] || []
    : [];

  const labeledOutputs = outputActivations.map((value, idx) => ({
    label: IRIS_CLASS_NAMES[idx] ?? `Class ${idx + 1}`,
    value,
    idx
  }));

  const passMetrics = useMemo<MetricsPoint[]>(() => {
    if (snapshots.length === 0) {
      return [];
    }

    let cumulativeLoss = 0;
    let cumulativeCorrect = 0;

    return snapshots.map((snapshot, index) => {
      cumulativeLoss += snapshot.loss;
      if (snapshot.prediction === snapshot.actualClass) {
        cumulativeCorrect += 1;
      }
      const passes = index + 1;
      return {
        id: snapshot.step,
        avgLoss: cumulativeLoss / passes,
        accuracy: cumulativeCorrect / passes,
        label: passes.toString()
      };
    });
  }, [snapshots]);

  const trainSamples = useMemo(() => trainingManager ? trainingManager.getTrainData() : [], [trainingManager]);

  const vizWidth = Math.max((isMobile ? viewportWidth - 32 : 900), 320);
  const vizHeight = isMobile ? 420 : 500;
  const metricsWidth = Math.max((isMobile ? viewportWidth - 32 : 860), 320);
  const metricsHeight = isMobile ? 220 : 260;
  const scatterWidth = isMobile ? vizWidth : 240;
  const scatterHeight = isMobile ? 200 : 220;

  const controlWrapperStyle: CSSProperties = {
    marginBottom: isMobile ? "28px" : "32px",
    position: isMobile ? "sticky" : "static",
    top: isMobile ? "16px" : undefined,
    zIndex: isMobile ? 30 : undefined,
    background: isMobile ? "rgba(249, 250, 251, 0.95)" : "transparent",
    backdropFilter: isMobile ? "blur(8px)" : undefined,
    borderRadius: isMobile ? "16px" : undefined,
    padding: isMobile ? "12px" : undefined,
    boxShadow: isMobile ? "0 12px 24px rgba(15, 23, 42, 0.12)" : "none"
  };

  const buttonRowStyle: CSSProperties = {
    display: "flex",
    gap: isMobile ? "12px" : "16px",
    alignItems: isMobile ? "stretch" : "center",
    flexWrap: "wrap",
    flexDirection: isMobile ? "column" : "row",
    width: "100%"
  };

  const statusBadgeStyle: CSSProperties = {
    marginTop: isMobile ? "10px" : "12px",
    background: "#e0e7ff",
    border: "1px solid #c7d2fe",
    color: "#1d4ed8",
    fontSize: "13px",
    borderRadius: "10px",
    padding: "10px 14px",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    boxShadow: "0 8px 16px rgba(59, 130, 246, 0.15)"
  };

  if (isLoading) {
    return (
      <div style={{
        padding: isMobile ? "24px" : "40px",
        maxWidth: isMobile ? "100%" : "1200px",
        margin: "0 auto",
        fontFamily: "system-ui, -apple-system, sans-serif"
      }}>
        <h1>Loading Iris Dataset...</h1>
        <p>Check the console for logs 🔍</p>
      </div>
    );
  }

  return (
    <div style={{
      padding: isMobile ? "24px" : "40px",
      maxWidth: isMobile ? "100%" : "1200px",
      margin: "0 auto",
      fontFamily: "system-ui, -apple-system, sans-serif"
    }}>
      <h1 style={{
        fontSize: isMobile ? "24px" : "32px",
        fontWeight: "bold",
        marginBottom: isMobile ? "4px" : "8px",
        color: "#111827"
      }}>
        Neural Network Training Visualizer
      </h1>
      <p style={{
        fontSize: isMobile ? "14px" : "16px",
        color: "#6b7280",
        marginBottom: isMobile ? "20px" : "32px"
      }}>
        Training on Iris Dataset: 4 inputs → 8 → 8 → 3 outputs
      </p>

      {/* Controls */}
      <div style={controlWrapperStyle}>
        <div style={buttonRowStyle}>
          <button
            onClick={handleStartTraining}
            disabled={isTraining}
            style={{
              padding: "12px 24px",
              fontSize: "16px",
              fontWeight: "600",
              color: "white",
              background: isTraining ? "#9ca3af" : "#3b82f6",
              border: "none",
              borderRadius: "8px",
              cursor: isTraining ? "not-allowed" : "pointer",
              transition: "background 0.2s",
              width: isMobile ? "100%" : "auto"
            }}
          >
            {isTraining ? "Training..." : "Start Training"}
          </button>

          {isTraining && (
            <button
              onClick={isPaused ? handleResumeTraining : handlePauseTraining}
              disabled={isAnimating}
              style={{
                padding: "12px 24px",
                fontSize: "16px",
                fontWeight: "600",
                color: "white",
                background: isPaused ? "#10b981" : "#f97316",
                border: "none",
                borderRadius: "8px",
                cursor: isAnimating ? "not-allowed" : "pointer",
                transition: "background 0.2s",
                width: isMobile ? "100%" : "auto"
              }}
            >
              {isPaused ? "Resume Training" : "Pause Training"}
            </button>
          )}

          {snapshots.length > 0 && (
            <>
              <button
                onClick={handlePrevEpoch}
                disabled={currentEpoch === 0}
                style={{
                  padding: "12px 24px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "#374151",
                  background: currentEpoch === 0 ? "#e5e7eb" : "white",
                  border: "2px solid #d1d5db",
                  borderRadius: "8px",
                  cursor: currentEpoch === 0 ? "not-allowed" : "pointer",
                  width: isMobile ? "100%" : "auto"
                }}
              >
                ← Previous Epoch
              </button>

              <button
                onClick={handlePlayAnimation}
                disabled={isAnimating}
                style={{
                  padding: "12px 24px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "white",
                  background: isAnimating ? "#9ca3af" : "#10b981",
                  border: "none",
                  borderRadius: "8px",
                  cursor: isAnimating ? "not-allowed" : "pointer",
                  width: isMobile ? "100%" : "auto"
                }}
              >
                {isAnimating ? "Animating..." : "▶️ Animate Epoch"}
              </button>

              <button
                onClick={handleNextEpoch}
                disabled={currentEpoch === maxEpochs - 1}
                style={{
                  padding: "12px 24px",
                  fontSize: "16px",
                  fontWeight: "600",
                  color: "#374151",
                  background: currentEpoch === maxEpochs - 1 ? "#e5e7eb" : "white",
                  border: "2px solid #d1d5db",
                  borderRadius: "8px",
                  cursor: currentEpoch === maxEpochs - 1 ? "not-allowed" : "pointer",
                  width: isMobile ? "100%" : "auto"
                }}
              >
                Next Epoch →
              </button>

              <div style={{
                fontSize: "14px",
                color: "#6b7280",
                alignSelf: isMobile ? "flex-start" : "center"
              }}>
                Epoch {currentEpoch + 1} / {maxEpochs}
              </div>
            </>
          )}
        </div>

        {statusMessage && (
          <div style={statusBadgeStyle}>
            <span role="img" aria-hidden="true">ℹ️</span>
            <span>{statusMessage}</span>
          </div>
        )}
      </div>

      {/* Visualization */}
      {trainingManager && (
        <div style={{
          border: "2px dashed #e5e7eb",
          borderRadius: "8px",
          padding: isMobile ? "20px" : "40px",
          textAlign: "center",
          background: "#f9fafb",
          overflowX: "auto"
        }}>
          <div style={{
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            gap: isMobile ? "16px" : "24px",
            alignItems: "stretch"
          }}>
            <div style={{ position: "relative", flex: "1 1 auto", minWidth: 0 }}>
              <NetworkVisualizer
                mlp={trainingManager.getNetwork()}
                inputs={trainSamples[0]?.inputs || []}
                snapshot={getCurrentSnapshot()}
                width={vizWidth}
                height={vizHeight}
                animate={isAnimating}
                onAnimationComplete={handleAnimationComplete}
                key={`${currentEpoch}-${animationTrigger}`}
                epochSnapshots={getCurrentEpochSnapshots()}
              />
              <LegendOverlay />
            </div>
            {trainSamples.length > 0 && (
              <div style={{
                flex: isMobile ? "0 0 auto" : "0 0 240px",
                display: "flex",
                justifyContent: "center"
              }}>
                <FeatureScatter
                  samples={trainSamples}
                  currentSample={currentSnapshot?.sample}
                  width={scatterWidth}
                  height={scatterHeight}
                />
              </div>
            )}
          </div>


          {passMetrics.length > 0 && (
            <div style={{
              marginTop: isMobile ? "24px" : "32px",
              padding: isMobile ? "12px" : "16px",
              background: "#ffffff",
              borderRadius: "8px",
              border: "1px solid #e5e7eb",
              overflowX: "auto"
            }}>
              <div style={{
                fontWeight: 600,
                fontSize: isMobile ? "14px" : "15px",
                color: "#111827",
                marginBottom: "12px"
              }}>
                Training Metrics by Pass
              </div>
              <div style={{ display: "flex", justifyContent: "center", minWidth: metricsWidth }}>
                <MetricsChart metrics={passMetrics} width={metricsWidth} height={metricsHeight} />
              </div>
            </div>
          )}

          {snapshots.length > 0 && currentSnapshot && (
            <>
              <div style={{
                marginTop: isMobile ? "16px" : "20px",
                padding: isMobile ? "12px" : "16px",
                background: "#ffffff",
                borderRadius: "8px",
                textAlign: "left",
                fontSize: isMobile ? "13px" : "14px",
                color: "#374151"
              }}>
                <div style={{ fontWeight: "600", marginBottom: "8px" }}>
                  Epoch {currentEpoch + 1} Summary:
                </div>
                <div>Training Samples: {currentEpochSnapshots.length}</div>
                <div>Average Loss: {currentEpochAvgLoss.toFixed(4)}</div>
                <div>Accuracy: {currentEpochAccuracy.toFixed(1)}%</div>
              </div>

              <div style={{
                marginTop: "16px",
                padding: isMobile ? "12px" : "16px",
                background: "#ffffff",
                borderRadius: "8px",
                textAlign: "left",
                fontSize: isMobile ? "13px" : "14px",
                color: "#374151",
                display: "grid",
                gap: isMobile ? "12px" : "16px"
              }}>
                <div style={{ fontWeight: 600 }}>Current Sample Details</div>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: isMobile ? "1fr" : "repeat(auto-fit, minmax(200px, 1fr))",
                  gap: isMobile ? "12px" : "16px"
                }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    <div style={{ fontSize: isMobile ? "12px" : "13px", fontWeight: 600, color: "#111827" }}>
                      Inputs
                    </div>
                    {labeledInputs.map(({ label, value }) => (
                      <div
                        key={label}
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          padding: isMobile ? "8px 10px" : "8px 12px",
                          background: "#f9fafb",
                          borderRadius: "6px",
                          border: "1px solid #e5e7eb",
                          fontSize: isMobile ? "12px" : "13px"
                        }}
                      >
                        <span>{label}</span>
                        <span style={{ fontVariantNumeric: "tabular-nums" }}>{value.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    <div style={{ fontSize: isMobile ? "12px" : "13px", fontWeight: 600, color: "#111827" }}>
                      Output predictions
                    </div>
                    {labeledOutputs.map(({ label, value, idx }) => {
                      const isPrediction = currentSnapshot.prediction === idx;
                      const isTarget = currentSnapshot.actualClass === idx;
                      const background = isPrediction && isTarget
                        ? "#dcfce7"
                        : isPrediction
                          ? "#dbeafe"
                          : isTarget
                            ? "#fef9c3"
                            : "#f9fafb";
                      const border = isPrediction || isTarget ? "#94a3b8" : "#e5e7eb";

                      return (
                        <div
                          key={label}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            padding: isMobile ? "8px 10px" : "8px 12px",
                            background,
                            borderRadius: "6px",
                            border: `1px solid ${border}`,
                            fontSize: isMobile ? "12px" : "13px",
                            fontWeight: isPrediction ? 600 : 500,
                            color: "#111827",
                            fontVariantNumeric: "tabular-nums"
                          }}
                        >
                          <span>
                            {label}
                            {isTarget ? " (target)" : ""}
                            {isPrediction && !isTarget ? " (prediction)" : ""}
                            {isPrediction && isTarget ? " (correct)" : ""}
                          </span>
                          <span>{value.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div style={{ fontSize: isMobile ? "12px" : "13px", color: "#4b5563" }}>
                  Predicted class: <strong>{IRIS_CLASS_NAMES[currentSnapshot.prediction] ?? "Unknown"}</strong> • Actual class: <strong>{IRIS_CLASS_NAMES[currentSnapshot.actualClass] ?? currentSnapshot.sample.label}</strong>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default Container;
