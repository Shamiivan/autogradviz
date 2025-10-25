import { Value } from "../micrograd/engine";
import { MLP } from "../micrograd/nn";
import type { NetworkData, NodeData, EdgeData, LayerData } from "./types";

/**
 * Extracts the network structure from an MLP and input values
 * Creates a graph representation suitable for D3 visualization
 */
export function extractNetworkData(mlp: MLP, inputs: Value[]): NetworkData {
  const nodes: NodeData[] = [];
  const edges: EdgeData[] = [];
  const layers: LayerData[] = [];

  // Layer 0: Input nodes
  const inputNodes = inputs.map((val, i) => ({
    id: `input-${i}`,
    value: val,
    layer: 0,
    indexInLayer: i,
    type: 'input' as const
  }));
  nodes.push(...inputNodes);
  layers.push({ layer: 0, neurons: [], size: inputs.length });

  // Track previous layer nodes for connecting edges
  let prevLayerNodes = inputNodes;

  // Process each layer in the MLP
  mlp.layers.forEach((layer, layerIdx) => {
    const layerNum = layerIdx + 1;
    const currentLayerNodes: NodeData[] = [];
    const isOutputLayer = layerNum === mlp.layers.length;

    layer.neurons.forEach((neuron, neuronIdx) => {
      // Create node for this neuron's output
      const nodeId = `L${layerNum}-N${neuronIdx}`;
      const nodeData: NodeData = {
        id: nodeId,
        value: new Value(0), // Will be populated during forward pass
        layer: layerNum,
        indexInLayer: neuronIdx,
        type: isOutputLayer ? 'output' : 'hidden'
      };
      nodes.push(nodeData);
      currentLayerNodes.push(nodeData);

      // Create edges from previous layer nodes to this neuron
      // Each weight connects a previous node to this neuron
      neuron.w.forEach((weight, weightIdx) => {
        const sourceNode = prevLayerNodes[weightIdx];
        edges.push({
          id: `${sourceNode.id}->${nodeId}`,
          source: sourceNode.id,
          target: nodeId,
          weight: weight
        });
      });

      // Optional: Visualize bias as a special node/edge
      // Uncomment if you want to show biases explicitly
      /*
      edges.push({
        id: `bias->${nodeId}`,
        source: `bias-L${layerNum}`,
        target: nodeId,
        weight: neuron.b
      });
      */
    });

    layers.push({
      layer: layerNum,
      neurons: layer.neurons,
      size: layer.neurons.length
    });
    prevLayerNodes = currentLayerNodes;
  });

  return { nodes, edges, layers };
}

/**
 * Updates node values after a forward pass
 * This syncs the visualization with actual computed values
 */
export function updateNodeValues(
  networkData: NetworkData,
  mlp: MLP,
  inputs: Value[]
): void {
  // Update input nodes
  inputs.forEach((val, i) => {
    const node = networkData.nodes.find(n => n.id === `input-${i}`);
    if (node) node.value = val;
  });

  // Compute forward pass and store intermediate values
  let currentOutputs: Value[] = inputs;

  mlp.layers.forEach((layer, layerIdx) => {
    const layerNum = layerIdx + 1;
    const outputs: Value[] = [];

    layer.neurons.forEach((neuron, neuronIdx) => {
      // Compute this neuron's output
      const output = neuron.forward(currentOutputs);
      outputs.push(output);

      // Update the corresponding node
      const nodeId = `L${layerNum}-N${neuronIdx}`;
      const node = networkData.nodes.find(n => n.id === nodeId);
      if (node) node.value = output;
    });

    currentOutputs = outputs;
  });
}