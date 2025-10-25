
import React from "react";
import { MLP } from "../micrograd/nn";
import { Value } from "../micrograd/engine";
import { NetworkVisualizer } from "./NetworkVizualiser";
import "../App.css";

function Container() {
  // Create MLP once and train it
  const mlp = React.useMemo(() => {
    const network = new MLP(2, [6, 6, 1]);

    const trainingData = [
      { x: [new Value(0), new Value(0)], y: new Value(0) },
      { x: [new Value(0), new Value(1)], y: new Value(1) },
      { x: [new Value(1), new Value(0)], y: new Value(1) },
      { x: [new Value(1), new Value(1)], y: new Value(0) },
    ];

    // Quick training for interesting weights
    for (let i = 0; i < 5; i++) {
      for (const { x, y } of trainingData) {
        network.zeroGrad();
        const pred = network.forward(x);
        const predValue = Array.isArray(pred) ? pred[0] : pred;

        // Compute loss (MSE) and backward
        predValue.data -= y.data;

        for (const p of network.parameters()) {
          p.data -= 0.01 * p.grad;
        }
      }
    }

    return network;
  }, []);

  const inputs = React.useMemo(() => [
    new Value(0.5),
    new Value(-0.3)
  ], []);

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
        Neural Network Architecture
      </h1>
      <p style={{
        fontSize: "16px",
        color: "#6b7280",
        marginBottom: "32px"
      }}>
        2 inputs → 6 → 6 → 1 output
      </p>

      <NetworkVisualizer
        mlp={mlp}
        inputs={inputs}
        width={900}
        height={500}
      />

      <div style={{
        marginTop: "24px",
        padding: "20px",
        background: "#f9fafb",
        borderRadius: "8px",
        fontSize: "14px"
      }}>
        <div style={{ fontWeight: "600", marginBottom: "8px" }}>Legend:</div>
        <div style={{ display: "flex", gap: "20px", flexWrap: "wrap" }}>
          <div><span style={{ color: "#10b981" }}>●</span> Input nodes</div>
          <div><span style={{ color: "#6366f1" }}>●</span> Hidden nodes</div>
          <div><span style={{ color: "#f59e0b" }}>●</span> Output nodes</div>
          <div><span style={{ color: "#3b82f6" }}>━</span> Positive weights</div>
          <div><span style={{ color: "#ef4444" }}>━</span> Red edges = Negative weights</div>
          <div>Thickness = weight magnitude</div>
        </div>
      </div>
    </div>
  );
}

export default Container;