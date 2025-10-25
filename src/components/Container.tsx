import React from "react";
import { MLP } from "../micrograd/nn";
import { Value } from "../micrograd/engine";
import { NetworkVisualizer } from "./NetworkVizualiser";
import "../App.css";

function Container() {
  // Create a simple neural network: 2 inputs, 4 hidden, 1 output
  const mlp = new MLP(2, [4, 1]);

  // Create sample inputs
  const inputs = [
    new Value(0.5),
    new Value(-0.3)
  ];

  // Optional: Train the network a bit so we have interesting values
  // This is just for demonstration
  React.useEffect(() => {
    // Simple XOR-like training
    const trainingData = [
      { x: [new Value(0), new Value(0)], y: new Value(0) },
      { x: [new Value(0), new Value(1)], y: new Value(1) },
      { x: [new Value(1), new Value(0)], y: new Value(1) },
      { x: [new Value(1), new Value(1)], y: new Value(0) },
    ];

    // Just a few iterations to get interesting weights
    for (let i = 0; i < 5; i++) {
      for (const { x, y } of trainingData) {
        mlp.zeroGrad();
        const pred = mlp.forward(x);
        const predValue = Array.isArray(pred) ? pred[0] : pred;

        // Simple squared error
        const loss = predValue.data - y.data;

        // Update parameters
        for (const p of mlp.parameters()) {
          p.data -= 0.01 * p.grad;
        }
      }
    }
  }, []);

  const handleStepChange = (step: any) => {
    console.log("Step changed:", step);
  };

  return (
    <div className="App" style={{
      padding: "40px",
      maxWidth: "1200px",
      margin: "0 auto",
      fontFamily: "system-ui, -apple-system, sans-serif"
    }}>
      <header style={{ marginBottom: "40px" }}>
        <h1 style={{
          fontSize: "32px",
          fontWeight: "bold",
          marginBottom: "8px",
          color: "#111827"
        }}>
          Micrograd Neural Network Visualizer
        </h1>
        <p style={{
          fontSize: "16px",
          color: "#6b7280"
        }}>
          Interactive visualization of forward and backward propagation through a neural network
        </p>
      </header>

      <div style={{
        background: "#f9fafb",
        padding: "24px",
        borderRadius: "12px",
        marginBottom: "24px"
      }}>
        <h2 style={{ fontSize: "18px", fontWeight: "600", marginBottom: "12px" }}>
          Network Architecture
        </h2>
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          <li style={{ marginBottom: "4px" }}>
            <strong>Input Layer:</strong> 2 neurons
          </li>
          <li style={{ marginBottom: "4px" }}>
            <strong>Hidden Layer:</strong> 4 neurons (sigmoid activation)
          </li>
          <li style={{ marginBottom: "4px" }}>
            <strong>Output Layer:</strong> 1 neuron (linear)
          </li>
        </ul>
      </div>

      <NetworkVisualizer
        mlp={mlp}
        inputs={inputs}
        width={900}
        height={500}
        onStepChange={handleStepChange}
      />

      <div style={{
        marginTop: "40px",
        padding: "24px",
        background: "#eff6ff",
        borderRadius: "12px",
        border: "1px solid #bfdbfe"
      }}>
        <h3 style={{ fontSize: "16px", fontWeight: "600", marginBottom: "12px" }}>
          How to use:
        </h3>
        <ol style={{ paddingLeft: "20px", margin: 0 }}>
          <li style={{ marginBottom: "8px" }}>
            Click <strong>"Play Forward"</strong> to animate the forward propagation through the network
          </li>
          <li style={{ marginBottom: "8px" }}>
            Use <strong>"Step Forward"</strong> and <strong>"Step Back"</strong> to move through the animation one step at a time
          </li>
          <li style={{ marginBottom: "8px" }}>
            Click <strong>"Play Backward"</strong> to see gradient backpropagation
          </li>
          <li style={{ marginBottom: "8px" }}>
            Toggle <strong>"Show Values"</strong> to display neuron activations
          </li>
          <li style={{ marginBottom: "8px" }}>
            Toggle <strong>"Show Gradients"</strong> during backward pass to see gradient values
          </li>
        </ol>
      </div>

      <div style={{
        marginTop: "24px",
        padding: "20px",
        background: "#fef3c7",
        borderRadius: "8px",
        fontSize: "14px"
      }}>
        <strong>Visual Legend:</strong>
        <ul style={{ marginTop: "8px", paddingLeft: "20px" }}>
          <li><span style={{ color: "#10b981" }}>●</span> Green = Input nodes</li>
          <li><span style={{ color: "#6366f1" }}>●</span> Indigo = Hidden nodes</li>
          <li><span style={{ color: "#f59e0b" }}>●</span> Amber = Output nodes</li>
          <li><span style={{ color: "#3b82f6" }}>━</span> Blue edges = Positive weights</li>
          <li><span style={{ color: "#ef4444" }}>━</span> Red edges = Negative weights</li>
          <li>Edge thickness = Weight magnitude</li>
        </ul>
      </div>
    </div>
  );
}

export default Container;