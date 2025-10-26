import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { MLP } from '../micrograd/nn';
import { Value } from '../micrograd/engine';
import type { TrainingSnapshot } from '../micrograd/types';
import { IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES } from '../utils';

interface NetworkVisualizerProps {
  mlp: MLP;
  inputs: Value[];
  snapshot?: TrainingSnapshot; // Optional snapshot for activation visualization
  width?: number;
  height?: number;
  animate?: boolean; // Trigger animation
  onAnimationComplete?: () => void; // Callback when animation finishes
  epochSnapshots?: TrainingSnapshot[]; // All snapshots for the current epoch
}

interface NodeData {
  id: string;
  x: number;
  y: number;
  type: 'input' | 'hidden' | 'output';
  activation?: number; // Activation value for coloring
  layerIndex: number;
  neuronIndex: number;
  label?: string;
}

interface EdgeData {
  source: NodeData;
  target: NodeData;
  weight: number;
}

const NODE_RADIUS = 18;
const MIN_NODE_SPACING = NODE_RADIUS * 2.6;
const MAX_NODE_SPACING = NODE_RADIUS * 5;

export function NetworkVisualizer({
  mlp,
  inputs,
  snapshot,
  width = 1000,
  height = 500,
  animate = false,
  onAnimationComplete,
  epochSnapshots = [],
}: NetworkVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const animationRef = useRef<boolean>(false);

  const layerSizes = [inputs.length, ...mlp.layers.map(layer => layer.neurons.length)];
  const maxLayerSize = layerSizes.length > 0 ? Math.max(...layerSizes) : 1;
  const baseVerticalPadding = NODE_RADIUS * 3;
  const minCanvasHeight = baseVerticalPadding * 2 + Math.max(maxLayerSize - 1, 0) * MIN_NODE_SPACING;
  const canvasHeight = Math.max(height, minCanvasHeight);

  const ACTIVATION_THRESHOLD = 0.5;

  // Gradient threshold - show gradient color if > 0.01
  const GRADIENT_THRESHOLD = 0.01;

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg
      .attr('height', canvasHeight)
      .attr('viewBox', `0 0 ${width} ${canvasHeight}`)
      .attr('preserveAspectRatio', 'xMidYMid meet');

    // Extract network structure
    const nodes: NodeData[] = [];
    const edges: EdgeData[] = [];

    const numLayers = mlp.layers.length + 1; // +1 for input layer
    const maxHorizontalPadding = Math.max(40, width / 2 - NODE_RADIUS * 2);
    const horizontalPadding = Math.min(Math.max(width * 0.12, 48), maxHorizontalPadding);
    const startX = horizontalPadding;
    const endX = Math.max(startX + 80, width - horizontalPadding);
    const layerX = d3.scaleLinear()
      .domain([0, numLayers - 1])
      .range([startX, endX]);

    const verticalPadding = Math.max(
      baseVerticalPadding,
      (canvasHeight - Math.max(maxLayerSize - 1, 0) * MIN_NODE_SPACING) / 2
    );
    const availableHeight = Math.max(canvasHeight - verticalPadding * 2, NODE_RADIUS * 2);
    const getNodeY = (layerSize: number, index: number) => {
      if (layerSize <= 1) {
        return canvasHeight / 2;
      }
      const spacing = Math.min(
        MAX_NODE_SPACING,
        Math.max(MIN_NODE_SPACING, availableHeight / (layerSize - 1))
      );
      const totalLayerHeight = spacing * (layerSize - 1);
      const startY = (canvasHeight - totalLayerHeight) / 2;
      return startY + index * spacing;
    };

    // Create enhanced color scale with more dramatic differences
    // FOR FORWARD PASS (ACTIVATION)
    const activationColorScale = (activation: number) => {
      if (activation < ACTIVATION_THRESHOLD) return 'white';
      if (activation < 0.5) return '#ef4444'; // red
      if (activation < 0.7) return '#eab308'; // yellow
      return '#10b981'; // green
    };

    const borderColorScale = (activation: number) => {
      if (activation < ACTIVATION_THRESHOLD) return '#9ca3af'; // gray
      if (activation < 0.5) return '#991b1b'; // dark red
      if (activation < 0.7) return '#a16207'; // dark yellow
      return '#047857'; // dark green
    };

    // FOR BACKWARD PASS (GRADIENTS) - Purple to Pink to White
    const gradientColorScale = (gradient: number) => {
      if (gradient < GRADIENT_THRESHOLD) return 'white';
      if (gradient < 0.5) return '#f9a8d4'; // light pink
      if (gradient < 1.0) return '#ec4899'; // hot pink
      return '#9333ea'; // deep purple
    };

    const gradientBorderScale = (gradient: number) => {
      if (gradient < GRADIENT_THRESHOLD) return '#9ca3af';
      if (gradient < 0.5) return '#be185d'; // dark pink
      if (gradient < 1.0) return '#be185d'; // dark pink
      return '#6b21a8'; // dark purple
    };

    // Input layer
    for (let i = 0; i < inputs.length; i++) {
      const layerSize = inputs.length;
      const y = getNodeY(layerSize, i);

      // Get activation from snapshot if available
      const activation = snapshot ? snapshot.activations[0][i] : undefined;

      nodes.push({
        id: `in-${i}`,
        x: layerX(0),
        y,
        type: 'input',
        activation,
        layerIndex: 0,
        neuronIndex: i,
        label: IRIS_FEATURE_NAMES[i] ?? `Feature ${i + 1}`
      });
    }

    // Hidden and output layers
    let prevNodes = nodes.slice();
    mlp.layers.forEach((layer, layerIdx) => {
      const layerNum = layerIdx + 1;
      const currentNodes: NodeData[] = [];
      const isOutput = layerNum === numLayers - 1;

      layer.neurons.forEach((neuron, neuronIdx) => {
        const layerSize = layer.neurons.length;
        const y = getNodeY(layerSize, neuronIdx);
        const nodeId = `L${layerNum}-${neuronIdx}`;

        // Get activation from snapshot if available
        const activation = snapshot ? snapshot.activations[layerNum][neuronIdx] : undefined;

        const targetNode: NodeData = {
          id: nodeId,
          x: layerX(layerNum),
          y,
          type: isOutput ? 'output' : 'hidden',
          activation,
          layerIndex: layerNum,
          neuronIndex: neuronIdx,
          label: isOutput ? (IRIS_CLASS_NAMES[neuronIdx] ?? `Class ${neuronIdx + 1}`) : undefined
        };

        currentNodes.push(targetNode);

        // Create edges
        neuron.w.forEach((weight, wIdx) => {
          edges.push({
            source: prevNodes[wIdx],
            target: targetNode,
            weight: weight.data
          });
        });
      });

      prevNodes = currentNodes;
      nodes.push(...currentNodes);
    });

    const inputNodes = nodes.filter(n => n.type === 'input');
    const outputNodes = nodes.filter(n => n.type === 'output');
    const outputLayerIndex = numLayers - 1;

    const formatInputLabel = (node: NodeData, value?: number) => {
      const base = node.label ?? `Input ${node.neuronIndex + 1}`;
      if (value === undefined) {
        return base;
      }
      return `${base}: ${value.toFixed(2)}`;
    };

    const formatOutputLabel = (
      node: NodeData,
      value: number | undefined,
      snap?: TrainingSnapshot
    ) => {
      const pieces: string[] = [];
      pieces.push(node.label ?? `Output ${node.neuronIndex + 1}`);
      if (value !== undefined) {
        pieces.push(value.toFixed(2));
      }
      if (snap) {
        const isPrediction = snap.prediction === node.neuronIndex;
        const isTarget = snap.actualClass === node.neuronIndex;
        if (isPrediction && isTarget) {
          pieces.push("prediction ✓");
        } else {
          if (isPrediction) {
            pieces.push("prediction");
          }
          if (isTarget) {
            pieces.push("target");
          }
        }
      }
      return pieces.join(" • ");
    };

    // Calculate weight scale
    const maxWeight = Math.max(...edges.map(e => Math.abs(e.weight)));
    const widthScale = d3.scalePow()
      .exponent(1.5)
      .domain([0, maxWeight || 1])
      .range([0.3, 6]);

    // Draw edges
    svg.append('g')
      .selectAll('line')
      .data(edges)
      .enter()
      .append('line')
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y)
      .attr('stroke', d => d.weight >= 0 ? '#3b82f6' : '#ef4444')
      .attr('stroke-width', d => widthScale(Math.abs(d.weight)))
      .attr('opacity', 0.6)
      .attr('stroke-linecap', 'round')
      .attr('class', (d) => `edge-${d.source.id}-${d.target.id}`);

    // Node colors (used when no activation data available)
    const nodeColors = {
      input: '#10b981',
      hidden: '#6366f1',
      output: '#f59e0b'
    };

    // Draw nodes
    const nodeGroup = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', d => `node-${d.id}`);

    // Add circles
    nodeGroup
      .append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', NODE_RADIUS)
      .attr('fill', d => {
        // Start with white/empty if animating
        if (animate && !animationRef.current) {
          return 'white';
        }
        // Use threshold-based coloring
        if (d.activation !== undefined) {
          return activationColorScale(d.activation);
        }
        return 'white';
      })
      .attr('stroke', d => {
        // Use border colors based on activation
        if (d.activation !== undefined) {
          return borderColorScale(d.activation);
        }
        return nodeColors[d.type];
      })
      .attr('stroke-width', 2.5)
      .attr('class', 'node-circle');

    // Add tooltips
    nodeGroup
      .append('title')
      .text(d => {
        const layerName = d.type === 'input' ? 'Input' :
          d.type === 'output' ? 'Output' :
            'Hidden';
        const activation = d.activation !== undefined ?
          `\nActivation: ${d.activation.toFixed(4)}` : '';
        return `${layerName} Layer ${d.layerIndex}\nNeuron ${d.neuronIndex}${activation}`;
      });

    // Add text labels showing activation values (small, centered on node)
    nodeGroup
      .append('text')
      .attr('x', d => d.x)
      .attr('y', d => d.y)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '9px')
      .attr('font-weight', 'bold')
      .attr('fill', '#1f2937')
      .attr('pointer-events', 'none')
      .attr('class', 'node-label')
      .text(d => d.activation !== undefined ? d.activation.toFixed(2) : '')
      .style('opacity', 0);

    // Add descriptor labels for inputs (left side)
    nodeGroup
      .filter(d => d.type === 'input' && Boolean(d.label))
      .append('text')
      .attr('x', d => d.x - 28)
      .attr('y', d => d.y)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#1f2937')
      .attr('pointer-events', 'none')
      .attr('class', 'node-feature-label')
      .text(d => formatInputLabel(d, d.activation));

    // Add descriptor labels for outputs (right side)
    nodeGroup
      .filter(d => d.type === 'output')
      .append('text')
      .attr('x', d => d.x + 28)
      .attr('y', d => d.y)
      .attr('text-anchor', 'start')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#1f2937')
      .attr('pointer-events', 'none')
      .attr('class', 'node-output-label')
      .text(d => formatOutputLabel(
        d,
        snapshot ? snapshot.activations[outputLayerIndex][d.neuronIndex] : d.activation,
        snapshot
      ));

    const updateNodeDescriptorText = (snap: TrainingSnapshot) => {
      const inputActivations = snap.activations[0] || [];
      const outputActivations = snap.activations[outputLayerIndex] || [];

      inputNodes.forEach(node => {
        svg.select(`.node-${node.id}`).select('.node-feature-label')
          .text(formatInputLabel(node, inputActivations[node.neuronIndex]));
      });

      outputNodes.forEach(node => {
        svg.select(`.node-${node.id}`).select('.node-output-label')
          .text(formatOutputLabel(node, outputActivations[node.neuronIndex], snap));
      });
    };

    if (snapshot) {
      updateNodeDescriptorText(snapshot);
    }

    // Animation logic - TWO PHASE: Forward pass then Backward pass
    if (animate && epochSnapshots.length > 0 && !animationRef.current) {
      animationRef.current = true;

      // Animate each sample in the epoch
      const animateSample = (sampleIndex: number) => {
        if (sampleIndex >= epochSnapshots.length) {
          // All samples processed - animation complete
          animationRef.current = false;
          if (snapshot) {
            updateNodeDescriptorText(snapshot);
          }
          if (onAnimationComplete) {
            onAnimationComplete();
          }
          return;
        }

        const currentSnapshot = epochSnapshots[sampleIndex];

        updateNodeDescriptorText(currentSnapshot);

        // Group nodes by layer
        const layerGroups: NodeData[][] = [];
        for (let i = 0; i < numLayers; i++) {
          layerGroups.push(nodes.filter(n => n.layerIndex === i));
        }

        // PHASE 1: FORWARD PASS (Input → Output)
        const animateForwardLayer = (layerIndex: number, delay: number) => {
          if (layerIndex >= numLayers) {
            // Forward pass complete - start backward pass
            setTimeout(() => {
              animateBackwardLayer(numLayers - 1, 0);
            }, 500); // Brief pause before backward
            return;
          }

          const layerNodes = layerGroups[layerIndex];

          // Animate edges feeding into this layer
          if (layerIndex > 0) {
            const incomingEdges = edges.filter(e => e.target.layerIndex === layerIndex);

            incomingEdges.forEach(edge => {
              svg.select(`.edge-${edge.source.id}-${edge.target.id}`)
                .transition()
                .duration(150)
                .delay(delay)
                .attr('opacity', 0.9)
                .transition()
                .duration(150)
                .attr('opacity', 0.6);
            });
          }

          // Animate nodes - GROW effect for high activation
          layerNodes.forEach((node) => {
            const nodeCircle = svg.select(`.node-${node.id}`).select('.node-circle');
            const nodeLabel = svg.select(`.node-${node.id}`).select('.node-label');

            const activation = currentSnapshot.activations[layerIndex][node.neuronIndex];
            const fillColor = activationColorScale(activation);
            const strokeColor = borderColorScale(activation);

            // Scale based on activation - DRAMATIC size difference
            const scale = activation >= ACTIVATION_THRESHOLD ? 1 + (activation * 0.4) : 1.0;

            nodeCircle
              .transition()
              .duration(200)
              .delay(delay + 100)
              .attr('fill', fillColor)
              .attr('stroke', strokeColor)
              .attr('r', NODE_RADIUS * scale); // GROW for high activation

            // Show activation value
            nodeLabel
              .text(activation.toFixed(2))
              .transition()
              .duration(200)
              .delay(delay + 100)
              .style('opacity', 1);
          });

          setTimeout(() => {
            animateForwardLayer(layerIndex + 1, 0);
          }, delay + 300);
        };

        // PHASE 2: BACKWARD PASS (Output → Input) - REVERSE ORDER
        const animateBackwardLayer = (layerIndex: number, delay: number) => {
          if (layerIndex < 0) {
            // Backward pass complete - start weight update phase
            setTimeout(() => {
              animateWeightUpdates();
            }, 300);
            return;
          }

          const layerNodes = layerGroups[layerIndex];

          // Animate edges in REVERSE - orange color for gradients
          if (layerIndex < numLayers - 1) {
            const outgoingEdges = edges.filter(e => e.source.layerIndex === layerIndex);

            outgoingEdges.forEach(edge => {
              const baseStroke = edge.weight >= 0 ? '#3b82f6' : '#ef4444';
              const edgeSelection = svg.select(`.edge-${edge.source.id}-${edge.target.id}`);
              const targetLayerIdx = edge.target.layerIndex - 1;
              const targetGradient = targetLayerIdx >= 0
                ? Math.abs(currentSnapshot.gradients[targetLayerIdx]?.[edge.target.neuronIndex] ?? 0)
                : 0;
              const gradientWidth = targetGradient > 0
                ? 2.5 + Math.min(targetGradient, 1.5) * 2
                : widthScale(Math.abs(edge.weight));

              edgeSelection
                .attr('stroke-dasharray', '8 12')
                .attr('stroke-dashoffset', 0)
                .transition()
                .duration(150)
                .delay(delay)
                .attr('stroke', '#f97316')
                .attr('stroke-width', gradientWidth)
                .attr('opacity', 0.9)
                .transition()
                .duration(320)
                .ease(d3.easeLinear)
                .attr('stroke-dashoffset', -24)
                .transition()
                .duration(150)
                .attr('stroke', baseStroke)
                .attr('stroke-width', widthScale(Math.abs(edge.weight)))
                .attr('opacity', 0.6)
                .attr('stroke-dasharray', null)
                .attr('stroke-dashoffset', null);
            });
          }

          // Animate gradient visualization - SHRINK effect for high gradient
          layerNodes.forEach((node) => {
            const nodeCircle = svg.select(`.node-${node.id}`).select('.node-circle');
            const nodeLabel = svg.select(`.node-${node.id}`).select('.node-label');

            // Get gradient from snapshot (skip input layer as it has no gradients)
            const gradient = layerIndex > 0 ?
              (currentSnapshot.gradients[layerIndex - 1]?.[node.neuronIndex] || 0) : 0;

            const gradFillColor = gradientColorScale(gradient);
            const gradStrokeColor = gradientBorderScale(gradient);
            const gradStrokeWidth = gradient >= GRADIENT_THRESHOLD
              ? 2.5 + Math.min(gradient, 1.5) * 1.5
              : 2.5;

            // SHRINK for high gradient - opposite of forward pass
            const scale = gradient >= GRADIENT_THRESHOLD ? 1 - (Math.min(gradient, 1.0) * 0.3) : 1.0;

            nodeCircle
              .transition()
              .duration(200)
              .delay(delay + 100)
              .attr('fill', gradFillColor)
              .attr('stroke', gradStrokeColor)
              .attr('stroke-width', gradStrokeWidth)
              .attr('r', NODE_RADIUS * scale); // SHRINK for high gradient

            // Show gradient value
            if (gradient >= GRADIENT_THRESHOLD) {
              nodeLabel
                .text(gradient.toFixed(3))
                .attr('fill', '#6b21a8') // Purple text for gradients
                .transition()
                .duration(200)
                .delay(delay + 100)
                .style('opacity', 1);
            }
          });

          setTimeout(() => {
            animateBackwardLayer(layerIndex - 1, 0);
          }, delay + 300);
        };

        // PHASE 3: WEIGHT UPDATE - Show edges changing thickness
        const animateWeightUpdates = () => {
          // Get all edges and their weight deltas
          edges.forEach((edge) => {
            // Find weight delta from snapshot
            const targetLayer = edge.target.layerIndex;

            // Weight delta is stored in target layer
            const delta = currentSnapshot.weightDeltas[targetLayer - 1]?.[edge.target.neuronIndex]?.[edge.source.neuronIndex] || 0;

            const absDelta = Math.abs(delta);

            // Calculate new weight (approximate)
            const newWeight = edge.weight - delta; // negative because we subtracted the delta
            const newThickness = widthScale(Math.abs(newWeight));

            // Flash color based on delta direction
            const flashColor = delta > 0 ? '#10b981' : '#ef4444'; // Green for increase, red for decrease

            if (absDelta > 0.001) { // Only animate significant changes
              svg.select(`.edge-${edge.source.id}-${edge.target.id}`)
                // Flash the color
                .transition()
                .duration(200)
                .attr('stroke', flashColor)
                .attr('opacity', 0.9)
                // Morph thickness
                .transition()
                .duration(400)
                .attr('stroke-width', newThickness)
                .attr('stroke', () => newWeight >= 0 ? '#3b82f6' : '#ef4444')
                .attr('opacity', 0.7)
                // Reset opacity
                .transition()
                .duration(200)
                .attr('opacity', 0.6)
                .attr('stroke-dasharray', null)
                .attr('stroke-dashoffset', null);
            } else {
              svg.select(`.edge-${edge.source.id}-${edge.target.id}`)
                .attr('stroke', newWeight >= 0 ? '#3b82f6' : '#ef4444')
                .attr('stroke-width', newThickness)
                .attr('opacity', 0.6)
                .attr('stroke-dasharray', null)
                .attr('stroke-dashoffset', null);
            }
          });

          // After weight updates, reset nodes and move to next sample
          setTimeout(() => {
            nodes.forEach(node => {
              svg.select(`.node-${node.id}`).select('.node-circle')
                .transition()
                .duration(150)
                .attr('fill', 'white')
                .attr('stroke', '#9ca3af')
                .attr('stroke-width', 2.5)
                .attr('r', NODE_RADIUS); // Reset size

              svg.select(`.node-${node.id}`).select('.node-label')
                .transition()
                .duration(150)
                .style('opacity', 0);
            });

            setTimeout(() => {
              animateSample(sampleIndex + 1);
            }, 200);
          }, 800); // Wait for weight animations to complete
        };

        // Start with forward pass
        animateForwardLayer(0, 0);
      };

      // Start with first sample
      animateSample(0);
    }

  }, [mlp, inputs, snapshot, width, canvasHeight, animate, onAnimationComplete, epochSnapshots, baseVerticalPadding, maxLayerSize]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={canvasHeight}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        background: '#ffffff'
      }}
    />
  );
}
