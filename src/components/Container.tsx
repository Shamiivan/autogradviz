import { useState, useEffect } from "react";
import { TrainingManager } from "../micrograd/training";
import type { TrainingSnapshot } from "../micrograd/types";
import { NetworkVisualizer } from "./NetworkVizualiser";

function Container() {
  const [trainingManager, setTrainingManager] = useState<TrainingManager | null>(null);
  const [snapshots, setSnapshots] = useState<TrainingSnapshot[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Initialize training manager and load data
  useEffect(() => {
    const initTraining = async () => {
      console.log("🎬 Initializing training system...");

      // Create training manager with config
      const manager = new TrainingManager({
        epochs: 5,
        learningRate: 0.1,
        trainTestSplit: 0.8
      });

      // Load Iris data
      await manager.loadData();

      setTrainingManager(manager);
      setIsLoading(false);

      console.log("✅ System ready! Click 'Start Training' to begin.");
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

    console.log("=".repeat(50) + "\n");

    // Log summary
    console.log("📈 Training Summary:");
    console.log(`   Total steps: ${allSnapshots.length}`);
    console.log(`   Initial loss: ${allSnapshots[0]?.loss.toFixed(4)}`);
    console.log(`   Final loss: ${allSnapshots[allSnapshots.length - 1]?.loss.toFixed(4)}`);

    // Log first few snapshots as examples
    console.log("\n📸 Sample Snapshots:");
    allSnapshots.slice(0, 3).forEach(snap => {
      console.log(`   Step ${snap.step}: Loss=${snap.loss.toFixed(4)}, ` +
        `Pred=${snap.prediction}, Actual=${snap.actualClass}, ` +
        `Label=${snap.sample.label}`);
    });

    setIsTraining(false);
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
        alignItems: "center"
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
          <div style={{
            fontSize: "14px",
            color: "#6b7280"
          }}>
            ✅ Captured {snapshots.length} snapshots
          </div>
        )}
      </div>

      {/* Visualization Placeholder */}
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
            width={900}
            height={500}
          />
        </div>
      )}

      {/* Instructions */}
      <div style={{
        marginTop: "32px",
        padding: "20px",
        background: "#eff6ff",
        borderRadius: "8px",
        border: "1px solid #bfdbfe"
      }}>
        <div style={{ fontWeight: "600", marginBottom: "8px", color: "#1e40af" }}>
          📋 Instructions:
        </div>
        <ol style={{ margin: 0, paddingLeft: "20px", color: "#1e40af" }}>
          <li>Open your browser's Developer Console (F12)</li>
          <li>Click "Start Training" to begin</li>
          <li>Watch the console for detailed training logs</li>
          <li>Next increment will add visualization animation! 🎨</li>
        </ol>
      </div>
    </div>
  );
}

export default Container;