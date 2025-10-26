import type { IrisSample } from "./preprocessor";

/**
 * Snapshot of network state at a specific training step
 */
export interface TrainingSnapshot {
  step: number;
  epoch: number;
  sampleIndex: number;
  sample: IrisSample;

  // Forward pass data
  activations: number[][]; // [layer][neuron] - activation values

  // Backward pass data
  gradients: number[][]; // [layer][neuron] - gradient values

  // Weight update data
  weightDeltas: number[][][]; // [layer][neuron][weight] - change in weights

  // Metrics
  loss: number;
  prediction: number; // predicted class index
  actualClass: number; // actual class index
}

/**
 * Training configuration
 */
export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  trainTestSplit: number; // e.g., 0.8 for 80% train
}