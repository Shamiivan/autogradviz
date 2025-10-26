import { useState, useEffect } from "react";
import { TrainingManager } from "../micrograd/training";
import type { TrainingSnapshot } from "../micrograd/types";
import { NetworkVisualizer } from "./NetworkVizualiser";

function Container() {
  const [trainingManager, setTrainingManager] = useState<TrainingManager | null>(null);
  const [snapshots, setSnapshots] = useState<TrainingSnapshot[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationTrigger, setAnimationTrigger] = useState(0);
  const [maxEpochs, setMaxEpochs] = useState(0);

  // Initialize training manager and load data
  useEffect(() => {
    const initTraining = async () => {
      console.log("Initializing training system...");

      // Create training manager with config - more epochs for better visualization
      const manager = new TrainingManager({
        epochs: 10,
        learningRate: 0.1,
        trainTestSplit: 0.8
      });

      // Load Iris data
      await manager.loadData();

      setTrainingManager(manager);
      setMaxEpochs(10);
      setIsLoading(false);

      console.log("System ready! Click 'Start Training' to begin.");
    };

    initTraining();
  }, []);

  // Start training
  const handleStartTraining = async () => {
    if (!trainingManager || isTraining) return;

    setIsTraining(true);
    console.log("\n" + "=".repeat(50));

    // Train the network
    await trainingManager.train();

    // Get all snapshots
    const allSnapshots = trainingManager.getSnapshots();
    setSnapshots(allSnapshots);
    setCurrentEpoch(0);

    console.log("=".repeat(50) + "\n");

    // Log summary
    console.log("Training Summary:");
    console.log(`   Total steps: ${allSnapshots.length}`);
    console.log(`   Initial loss: ${allSnapshots[0]?.loss.toFixed(4)}`);
    console.log(`   Final loss: ${allSnapshots[allSnapshots.length - 1]?.loss.toFixed(4)}`);

    setIsTraining(false);
  };

  // Play animation for current epoch
  const handlePlayAnimation = () => {
    if (snapshots.length === 0 || isAnimating) return;
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

  if (isLoading) {
    return (
      <div style={{
        padding: "40px",
        maxWidth: "1200px",
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
      padding: "40px",
      maxWidth: "1200px",
      margin: "0 auto",
      fontFamily: "system-ui, -apple-system, sans-serif"
    }}>
      <h1 style={{
        fontSize: "32px",
        fontWeight: "bold",
        marginBottom: "8px",
        color: "#111827"
      }}>
        Neural Network Training Visualizer
      </h1>
      <p style={{
        fontSize: "16px",
        color: "#6b7280",
        marginBottom: "32px"
      }}>
        Training on Iris Dataset: 4 inputs → 8 → 8 → 3 outputs
      </p>

      {/* Controls */}
      <div style={{
        marginBottom: "32px",
        display: "flex",
        gap: "16px",
        alignItems: "center",
        flexWrap: "wrap"
      }}>
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
            transition: "background 0.2s"
          }}
        >
          {isTraining ? "Training..." : "Start Training"}
        </button>

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
                cursor: currentEpoch === 0 ? "not-allowed" : "pointer"
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
                cursor: isAnimating ? "not-allowed" : "pointer"
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
                cursor: currentEpoch === maxEpochs - 1 ? "not-allowed" : "pointer"
              }}
            >
              Next Epoch →
            </button>

            <div style={{
              fontSize: "14px",
              color: "#6b7280"
            }}>
              Epoch {currentEpoch + 1} / {maxEpochs}
            </div>
          </>
        )}
      </div>

      {/* Visualization */}
      {trainingManager && (
        <div style={{
          border: "2px dashed #e5e7eb",
          borderRadius: "8px",
          padding: "40px",
          textAlign: "center",
          background: "#f9fafb"
        }}>
          <NetworkVisualizer
            mlp={trainingManager.getNetwork()}
            inputs={trainingManager.getTrainData()[0]?.inputs || []}
            snapshot={getCurrentSnapshot()}
            width={900}
            height={500}
            animate={isAnimating}
            onAnimationComplete={handleAnimationComplete}
            key={`${currentEpoch}-${animationTrigger}`}
            epochSnapshots={getCurrentEpochSnapshots()}
          />

          {/* Color Legend */}
          {snapshots.length > 0 && (
            <div style={{
              marginTop: "16px",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
              alignItems: "center"
            }}>
              {/* Forward Pass Legend */}
              <div style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                gap: "20px",
                fontSize: "13px",
                color: "#6b7280"
              }}>
                <div style={{ fontWeight: "600", color: "#3b82f6" }}>Forward Pass:</div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "white",
                    border: "2px solid #9ca3af"
                  }}></div>
                  <span>&lt; 0.3</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "#ef4444",
                    border: "2px solid #991b1b"
                  }}></div>
                  <span>0.3-0.5</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "#eab308",
                    border: "2px solid #a16207"
                  }}></div>
                  <span>0.5-0.7</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "24px",
                    height: "24px",
                    borderRadius: "50%",
                    background: "#10b981",
                    border: "2px solid #047857"
                  }}></div>
                  <span>&gt; 0.7 (Grows)</span>
                </div>
              </div>

              {/* Backward Pass Legend */}
              <div style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                gap: "20px",
                fontSize: "13px",
                color: "#6b7280"
              }}>
                <div style={{ fontWeight: "600", color: "#f97316" }}>Backward Pass:</div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "white",
                    border: "2px solid #9ca3af"
                  }}></div>
                  <span>&lt; 0.01</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "#f9a8d4",
                    border: "2px solid #be185d"
                  }}></div>
                  <span>0.01-0.5</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: "50%",
                    background: "#ec4899",
                    border: "2px solid #be185d"
                  }}></div>
                  <span>0.5-1.0</span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <div style={{
                    width: "16px",
                    height: "16px",
                    borderRadius: "50%",
                    background: "#9333ea",
                    border: "2px solid #6b21a8"
                  }}></div>
                  <span>&gt; 1.0 (Shrinks)</span>
                </div>
              </div>
            </div>
          )}

          {snapshots.length > 0 && getCurrentSnapshot() && (
            <div style={{
              marginTop: "20px",
              padding: "16px",
              background: "#ffffff",
              borderRadius: "8px",
              textAlign: "left",
              fontSize: "14px",
              color: "#374151"
            }}>
              <div style={{ fontWeight: "600", marginBottom: "8px" }}>
                Epoch {currentEpoch + 1} Summary:
              </div>
              <div>Training Samples: {getCurrentEpochSnapshots().length}</div>
              <div>Average Loss: {(getCurrentEpochSnapshots().reduce((sum, s) => sum + s.loss, 0) / getCurrentEpochSnapshots().length).toFixed(4)}</div>
              <div>Accuracy: {(getCurrentEpochSnapshots().filter(s => s.prediction === s.actualClass).length / getCurrentEpochSnapshots().length * 100).toFixed(1)}%</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Container;
