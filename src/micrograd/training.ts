import { MLP } from "./nn";
import { Value, add, mul } from "./engine";
import { loadIrisData, shuffle, splitTrainTest } from "./preprocessor";
import type { IrisSample } from "./preprocessor";
import type { TrainingSnapshot, TrainingConfig } from "./types";

/**
 * Manages training process and captures snapshots for visualization
 */
export class TrainingManager {
  private mlp: MLP;
  private trainData: IrisSample[] = [];
  private testData: IrisSample[] = [];
  private config: TrainingConfig;
  private snapshots: TrainingSnapshot[] = [];
  private currentStep = 0;

  constructor(config: TrainingConfig) {
    this.config = config;
    // Create MLP: 4 inputs (iris features) -> hidden layers -> 3 outputs (classes)
    this.mlp = new MLP(4, [8, 8, 3]);
    console.log("🧠 Neural Network created: 4 → 8 → 8 → 3");
  }

  /**
   * Load and split Iris dataset
   */
  async loadData(): Promise<void> {
    console.log("📊 Loading Iris dataset...");
    const allData = await loadIrisData();

    // Shuffle and split
    const shuffled = shuffle(allData);
    const { train, test } = splitTrainTest(shuffled, this.config.trainTestSplit);

    this.trainData = train;
    this.testData = test;

    console.log(`✅ Data loaded: ${train.length} training samples, ${test.length} test samples`);
    console.log(`   Sample labels: ${train.slice(0, 3).map(s => s.label).join(", ")}...`);
  }

  /**
   * Train for one epoch and capture snapshots
   */
  trainEpoch(epoch: number): void {
    console.log(`\n🔄 Epoch ${epoch + 1}/${this.config.epochs}`);

    let epochLoss = 0;

    for (let i = 0; i < this.trainData.length; i++) {
      const sample = this.trainData[i];

      // Capture snapshot before training step (forward pass)
      const snapshot = this.captureSnapshot(epoch, i, sample);

      epochLoss += snapshot.loss;
      this.currentStep++;

      // Perform gradient descent
      this.mlp.zeroGrad();

      // Forward pass (already done in captureSnapshot)
      const outputs = this.mlp.forward(sample.inputs);
      const predictions = Array.isArray(outputs) ? outputs : [outputs];

      // Compute loss (MSE)
      let loss = new Value(0);
      for (let j = 0; j < predictions.length; j++) {
        const diff = add(predictions[j], mul(sample.outputs[j], -1));
        loss = add(loss, mul(diff, diff));
      }

      // Backward pass
      loss.backward();

      // NOW capture gradients after backward pass
      this.captureGradients(snapshot);

      // Capture weights BEFORE update
      const oldWeights: number[][][] = this.mlp.layers.map(layer =>
        layer.neurons.map(neuron => neuron.w.map(w => w.data))
      );

      // Update weights
      for (const param of this.mlp.parameters()) {
        param.data -= this.config.learningRate * param.grad;
      }

      // Capture weight deltas AFTER update
      this.mlp.layers.forEach((layer, layerIdx) => {
        layer.neurons.forEach((neuron, neuronIdx) => {
          neuron.w.forEach((weight, weightIdx) => {
            const oldWeight = oldWeights[layerIdx][neuronIdx][weightIdx];
            const delta = weight.data - oldWeight;
            snapshot.weightDeltas[layerIdx][neuronIdx][weightIdx] = delta;
          });
        });
      });

      // Add snapshot AFTER we have gradients and weight deltas
      this.snapshots.push(snapshot);
    }

    const avgLoss = epochLoss / this.trainData.length;
    console.log(`   Average Loss: ${avgLoss.toFixed(4)}`);
  }

  /**
   * Capture a snapshot of the current network state
   */
  private captureSnapshot(epoch: number, sampleIndex: number, sample: IrisSample): TrainingSnapshot {
    // Forward pass
    const outputs = this.mlp.forward(sample.inputs);
    const predictions = Array.isArray(outputs) ? outputs : [outputs];

    // Capture activations from each layer
    const activations: number[][] = [];

    // Input layer activations
    activations.push(sample.inputs.map(v => v.data));

    // Hidden and output layer activations
    this.mlp.layers.forEach(layer => {
      const layerActivations = layer.neurons.map(neuron => {
        const output = neuron.forward(
          activations[activations.length - 1].map(v => new Value(v))
        );
        return Array.isArray(output) ? output[0].data : output.data;
      });
      activations.push(layerActivations);
    });

    // Compute loss for this sample
    let loss = 0;
    for (let i = 0; i < predictions.length; i++) {
      const diff = predictions[i].data - sample.outputs[i].data;
      loss += diff * diff;
    }

    // Get predicted class (argmax)
    const predictionValues = predictions.map(p => p.data);
    const predictedClass = predictionValues.indexOf(Math.max(...predictionValues));

    // Get actual class
    const actualClass = sample.outputs.findIndex(o => o.data === 1);

    // Placeholder gradients - will be filled after backward pass
    const gradients: number[][] = this.mlp.layers.map(layer =>
      layer.neurons.map(() => 0)
    );

    // Weight deltas (will be computed after update - for now empty)
    const weightDeltas: number[][][] = this.mlp.layers.map(layer =>
      layer.neurons.map(neuron => neuron.w.map(() => 0))
    );

    return {
      step: this.currentStep,
      epoch,
      sampleIndex,
      sample,
      activations,
      gradients,
      weightDeltas,
      loss,
      prediction: predictedClass,
      actualClass
    };
  }

  /**
   * Capture gradients after backward pass
   */
  private captureGradients(snapshot: TrainingSnapshot): void {
    // Capture gradients from each layer's neurons
    this.mlp.layers.forEach((layer, layerIdx) => {
      layer.neurons.forEach((neuron, neuronIdx) => {
        // Use the bias gradient as representative of neuron's gradient magnitude
        snapshot.gradients[layerIdx][neuronIdx] = Math.abs(neuron.b.grad);
      });
    });
  }

  /**
   * Run full training
   */
  async train(): Promise<void> {
    console.log("\n🚀 Starting training...");
    console.log(`   Config: ${this.config.epochs} epochs, lr=${this.config.learningRate}`);

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      this.trainEpoch(epoch);
    }

    console.log("\n✅ Training complete!");
    console.log(`   Total snapshots captured: ${this.snapshots.length}`);
  }

  /**
   * Get all captured snapshots
   */
  getSnapshots(): TrainingSnapshot[] {
    return this.snapshots;
  }

  /**
   * Get the trained network
   */
  getNetwork(): MLP {
    return this.mlp;
  }

  /**
   * Get training data
   */
  getTrainData(): IrisSample[] {
    return this.trainData;
  }

  /**
   * Get test data
   */
  getTestData(): IrisSample[] {
    return this.testData;
  }
}