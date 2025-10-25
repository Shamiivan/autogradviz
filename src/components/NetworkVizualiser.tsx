import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { MLP } from '../micrograd/nn';
import { Value } from '../micrograd/engine';

interface NetworkVisualizerProps {
  mlp: MLP;
  inputs: Value[];
  width?: number;
  height?: number;
}

interface NodeData {
  id: string;
  x: number;
  y: number;
  type: 'input' | 'hidden' | 'output';
}

interface EdgeData {
  source: NodeData;
  target: NodeData;
  weight: number;
}

export function NetworkVisualizer({
  mlp,
  inputs,
  width = 900,
  height = 500,
}: NetworkVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Extract network structure
    const nodes: NodeData[] = [];
    const edges: EdgeData[] = [];

    const numLayers = mlp.layers.length + 1; // +1 for input layer
    const layerX = d3.scaleLinear()
      .domain([0, numLayers - 1])
      .range([80, width - 80]);

    for (let i = 0; i < inputs.length; i++) {
      const layerSize = inputs.length;
      const y = height / 2 + (i - (layerSize - 1) / 2) * 60;
      nodes.push({
        id: `in-${i}`,
        x: layerX(0),
        y,
        type: 'input'
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
        const y = height / 2 + (neuronIdx - (layerSize - 1) / 2) * 60;
        const nodeId = `L${layerNum}-${neuronIdx}`;

        const targetNode: NodeData = {
          id: nodeId,
          x: layerX(layerNum),
          y,
          type: isOutput ? 'output' : 'hidden'
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

    // Calculate weight scale
    const maxWeight = Math.max(...edges.map(e => Math.abs(e.weight)));
    const widthScale = d3.scaleLinear()
      .domain([0, maxWeight])
      .range([0.5, 4]);

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
      .attr('stroke-linecap', 'round');

    // Draw nodes
    const nodeColors = {
      input: '#10b981',
      hidden: '#6366f1',
      output: '#f59e0b'
    };

    svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 18)
      .attr('fill', 'white')
      .attr('stroke', d => nodeColors[d.type])
      .attr('stroke-width', 2.5);

  }, [mlp, inputs, width, height]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      style={{
        border: '1px solid #e5e7eb',
        borderRadius: '8px',
        background: '#ffffff'
      }}
    />
  );
}