import type { Value } from '../engine';
import type { Neuron } from '../nn';

export interface NodeData {
  id: string;
  value: Value;           // Reference to actual Value object
  layer: number;
  indexInLayer: number;
  type: 'input' | 'hidden' | 'output';
  x?: number;             // Calculated by layout
  y?: number;
}

export interface EdgeData {
  id: string;
  source: string;         // Node id
  target: string;
  weight: Value;          // Reference to weight Value object
}

export interface LayerData {
  layer: number;
  neurons: Neuron[];
  size: number;
}

export interface NetworkData {
  nodes: NodeData[];
  edges: EdgeData[];
  layers: LayerData[];
}

export interface LayoutConfig {
  width: number;
  height: number;
  nodeRadius: number;
  layerSpacing: number;
  nodeSpacing: number;
}

export interface VizState {
  mode: 'architecture' | 'forward' | 'backward';
  showValues: boolean;
  showGradients: boolean;
  activeNodes: Set<string>;
  activeEdges: Set<string>;
}

export interface AnimationStep {
  type: 'forward' | 'backward';
  layerIndex: number;
  activeNodes: string[];
  activeEdges: string[];
  values: Map<string, number>;
  gradients?: Map<string, number>;
}