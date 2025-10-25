import type { AnimationStep, NetworkData, VizState } from "./types"
import type { MLP } from "../micrograd/nn";
import type { Value } from "../micrograd/engine";

/**
 * Animation controller for stepping through forward and backward passes
 */
export class AnimationController {
  private steps: AnimationStep[] = [];
  private currentStepIndex = 0;
  private isPlaying = false;
  private intervalId: number | null = null;

  constructor(
    private onStepChange: (step: AnimationStep, index: number, total: number) => void
  ) { }

  /**
   * Set the animation steps
   */
  setSteps(steps: AnimationStep[]) {
    this.steps = steps;
    this.currentStepIndex = 0;
  }

  /**
   * Get current step
   */
  getCurrentStep(): AnimationStep | null {
    return this.steps[this.currentStepIndex] || null;
  }

  /**
   * Play animation automatically
   */
  play(speed: number = 500) {
    if (this.isPlaying) return;
    this.isPlaying = true;

    this.intervalId = window.setInterval(() => {
      if (this.currentStepIndex >= this.steps.length - 1) {
        this.pause();
        return;
      }
      this.stepForward();
    }, speed);
  }

  /**
   * Pause animation
   */
  pause() {
    this.isPlaying = false;
    if (this.intervalId !== null) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Step forward one step
   */
  stepForward() {
    if (this.currentStepIndex < this.steps.length - 1) {
      this.currentStepIndex++;
      this.notifyStepChange();
    }
  }

  /**
   * Step backward one step
   */
  stepBackward() {
    if (this.currentStepIndex > 0) {
      this.currentStepIndex--;
      this.notifyStepChange();
    }
  }

  /**
   * Reset to beginning
   */
  reset() {
    this.pause();
    this.currentStepIndex = 0;
    if (this.steps.length > 0) {
      this.notifyStepChange();
    }
  }

  /**
   * Jump to specific step
   */
  jumpToStep(index: number) {
    if (index >= 0 && index < this.steps.length) {
      this.currentStepIndex = index;
      this.notifyStepChange();
    }
  }

  private notifyStepChange() {
    const step = this.steps[this.currentStepIndex];
    if (step) {
      this.onStepChange(step, this.currentStepIndex, this.steps.length);
    }
  }

  isAtStart(): boolean {
    return this.currentStepIndex === 0;
  }

  isAtEnd(): boolean {
    return this.currentStepIndex === this.steps.length - 1;
  }

  getProgress(): number {
    return this.steps.length > 0 ? this.currentStepIndex / (this.steps.length - 1) : 0;
  }
}

/**
 * Create animation steps for a forward pass through the network
 */
export function createForwardPassSteps(
  mlp: MLP,
  inputs: Value[],
  networkData: NetworkData
): AnimationStep[] {
  const steps: AnimationStep[] = [];

  // Step 1: Show input layer activation
  steps.push({
    type: 'forward',
    layerIndex: 0,
    activeNodes: networkData.nodes
      .filter(n => n.layer === 0)
      .map(n => n.id),
    activeEdges: [],
    values: new Map(inputs.map((val, i) => [`input-${i}`, val.data]))
  });

  // Process each layer
  let layerInputs = inputs;

  mlp.layers.forEach((layer, layerIdx) => {
    const layerNum = layerIdx + 1;
    const layerOutputs: Value[] = [];

    // For each neuron in the layer
    layer.neurons.forEach((neuron, neuronIdx) => {
      const nodeId = `L${layerNum}-N${neuronIdx}`;

      // Find incoming edges to this neuron
      const incomingEdges = networkData.edges.filter(e => e.target === nodeId);

      // Compute neuron output
      const output = neuron.forward(layerInputs);
      layerOutputs.push(output);

      // Create step showing this neuron's activation
      steps.push({
        type: 'forward',
        layerIndex: layerNum,
        activeNodes: [nodeId],
        activeEdges: incomingEdges.map(e => e.id),
        values: new Map([[nodeId, output.data]])
      });
    });

    // Step showing entire layer activated
    steps.push({
      type: 'forward',
      layerIndex: layerNum,
      activeNodes: networkData.nodes
        .filter(n => n.layer === layerNum)
        .map(n => n.id),
      activeEdges: [],
      values: new Map(
        layerOutputs.map((val, i) => [`L${layerNum}-N${i}`, val.data])
      )
    });

    layerInputs = layerOutputs;
  });

  return steps;
}

/**
 * Create animation steps for a backward pass through the network
 */
export function createBackwardPassSteps(
  mlp: MLP,
  networkData: NetworkData
): AnimationStep[] {
  const steps: AnimationStep[] = [];

  // Start from output layer and work backwards
  const numLayers = mlp.layers.length;

  // Initialize: show output gradient
  const outputLayer = numLayers;
  const outputNodes = networkData.nodes.filter(n => n.layer === outputLayer);

  steps.push({
    type: 'backward',
    layerIndex: outputLayer,
    activeNodes: outputNodes.map(n => n.id),
    activeEdges: [],
    values: new Map(),
    gradients: new Map(outputNodes.map(n => [n.id, n.value.grad]))
  });

  // Propagate backwards through each layer
  for (let layerIdx = numLayers - 1; layerIdx >= 0; layerIdx--) {
    const layerNum = layerIdx + 1;
    const layer = mlp.layers[layerIdx];

    // For each neuron in this layer
    layer.neurons.forEach((neuron, neuronIdx) => {
      const nodeId = `L${layerNum}-N${neuronIdx}`;

      // Find outgoing edges from this neuron
      const outgoingEdges = networkData.edges.filter(e => e.source === nodeId);

      // Get incoming edges (to show gradient flow)
      const incomingEdges = networkData.edges.filter(e => e.target === nodeId);

      // Create step showing gradient flow through this neuron
      steps.push({
        type: 'backward',
        layerIndex: layerNum,
        activeNodes: [nodeId],
        activeEdges: [...incomingEdges.map(e => e.id), ...outgoingEdges.map(e => e.id)],
        values: new Map(),
        gradients: new Map([[nodeId, neuron.w[0]?.grad || 0]])
      });
    });
  }

  return steps;
}

/**
 * Create a complete training step (forward + backward + update)
 */
export function createTrainingSteps(
  mlp: MLP,
  inputs: Value[],
  target: Value,
  learningRate: number,
  networkData: NetworkData
): AnimationStep[] {
  const steps: AnimationStep[] = [];

  // Forward pass steps
  steps.push(...createForwardPassSteps(mlp, inputs, networkData));

  // Compute loss and backward pass
  const output = mlp.forward(inputs);
  // Assuming single output for simplicity
  // You'd compute your loss here

  // Backward pass steps
  steps.push(...createBackwardPassSteps(mlp, networkData));

  // Parameter update step
  const allNodes = networkData.nodes.filter(n => n.layer > 0);
  steps.push({
    type: 'backward',
    layerIndex: -1, // Special marker for update step
    activeNodes: allNodes.map(n => n.id),
    activeEdges: networkData.edges.map(e => e.id),
    values: new Map(),
    gradients: new Map()
  });

  return steps;
}