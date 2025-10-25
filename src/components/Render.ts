import * as d3 from 'd3';
import type { NetworkData, VizState } from './types';

// Visual scales for mapping data to visual properties
export const scales = {
  // Weight width scale - thicker lines for larger weights
  weightWidth: d3.scaleLinear()
    .domain([0, 1])
    .range([0.5, 5])
    .clamp(true),

  // Weight color scale - red for negative, blue for positive
  weightColor: d3.scaleLinear<string>()
    .domain([-1, 0, 1])
    .range(['#ef4444', '#94a3b8', '#3b82f6'])
    .clamp(true),

  // Node activation color - for forward pass
  activationColor: d3.scaleLinear<string>()
    .domain([0, 0.5, 1])
    .range(['#ffffff', '#a5b4fc', '#4f46e5'])
    .clamp(true),

  // Gradient color scale - for backward pass
  gradientColor: d3.scaleSequential(d3.interpolateRdYlGn)
    .domain([-1, 1])
};

/**
 * Update weight scale domain based on actual weight values
 */
export function updateWeightScale(networkData: NetworkData) {
  const maxWeight = d3.max(networkData.edges, e => Math.abs(e.weight.data)) || 1;
  scales.weightWidth.domain([0, maxWeight]);
}

/**
 * Normalize a value to [0, 1] using sigmoid
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Main rendering function - renders the complete network
 */
export function renderNetwork(
  svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
  networkData: NetworkData,
  state: VizState
): void {
  // Update scale domains
  updateWeightScale(networkData);

  // Clear existing content
  svg.selectAll('*').remove();

  // Create layer groups (edges behind nodes)
  const edgesGroup = svg.append('g').attr('class', 'edges');
  const nodesGroup = svg.append('g').attr('class', 'nodes');

  // Render
  renderEdges(edgesGroup, networkData, state);
  renderNodes(nodesGroup, networkData, state);
}

/**
 * Render edges (connections between nodes)
 */
export function renderEdges(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  networkData: NetworkData,
  state: VizState
): void {
  const edges = group.selectAll<SVGLineElement, any>('line')
    .data(networkData.edges, (d: any) => d.id);

  // Enter + Update
  const edgeElements = edges.enter()
    .append('line')
    .merge(edges)
    .attr('x1', d => {
      const source = networkData.nodes.find(n => n.id === d.source);
      return source?.x || 0;
    })
    .attr('y1', d => {
      const source = networkData.nodes.find(n => n.id === d.source);
      return source?.y || 0;
    })
    .attr('x2', d => {
      const target = networkData.nodes.find(n => n.id === d.target);
      return target?.x || 0;
    })
    .attr('y2', d => {
      const target = networkData.nodes.find(n => n.id === d.target);
      return target?.y || 0;
    })
    .attr('stroke', d => {
      // Show gradients in backward mode
      if (state.mode === 'backward' && state.showGradients) {
        return scales.gradientColor(d.weight.grad);
      }
      // Show weights in forward/architecture mode
      return scales.weightColor(d.weight.data);
    })
    .attr('stroke-width', d => scales.weightWidth(Math.abs(d.weight.data)))
    .attr('opacity', d => {
      // Highlight active edges
      if (state.activeEdges.size === 0) return 0.6;
      return state.activeEdges.has(d.id) ? 1 : 0.2;
    })
    .attr('stroke-linecap', 'round');

  // Add transitions for smooth updates
  edgeElements.transition()
    .duration(300)
    .attr('stroke', d => {
      if (state.mode === 'backward' && state.showGradients) {
        return scales.gradientColor(d.weight.grad);
      }
      return scales.weightColor(d.weight.data);
    })
    .attr('stroke-width', d => scales.weightWidth(Math.abs(d.weight.data)));

  edges.exit().remove();
}

/**
 * Render nodes (neurons)
 */
export function renderNodes(
  group: d3.Selection<SVGGElement, unknown, null, undefined>,
  networkData: NetworkData,
  state: VizState
): void {
  // Bind data
  const nodeGroups = group.selectAll<SVGGElement, any>('g.node')
    .data(networkData.nodes, (d: any) => d.id);

  // Enter - create new node groups
  const enter = nodeGroups.enter()
    .append('g')
    .attr('class', 'node');

  // Add circle
  enter.append('circle')
    .attr('r', 20);

  // Add text label
  enter.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '0.35em')
    .style('font-size', '11px')
    .style('font-weight', '500')
    .style('pointer-events', 'none');

  // Update all nodes (enter + existing)
  const allNodes = enter.merge(nodeGroups);

  // Position node groups
  allNodes
    .attr('transform', d => `translate(${d.x}, ${d.y})`);

  // Update circles
  allNodes.select('circle')
    .attr('fill', d => {
      // Inactive nodes are white
      if (state.activeNodes.size > 0 && !state.activeNodes.has(d.id)) {
        return 'white';
      }
      // Forward mode: color by activation
      if (state.mode === 'forward') {
        return scales.activationColor(sigmoid(d.value.data));
      }
      // Backward mode: color by gradient
      if (state.mode === 'backward' && state.showGradients) {
        return scales.gradientColor(d.value.grad);
      }
      return 'white';
    })
    .attr('stroke', d => {
      // Different colors for different layer types
      if (d.type === 'input') return '#10b981';  // green
      if (d.type === 'output') return '#f59e0b'; // amber
      return '#6366f1'; // indigo for hidden
    })
    .attr('stroke-width', d => {
      // Thicker stroke for active nodes
      return state.activeNodes.has(d.id) ? 3 : 1.5;
    });

  // Update text
  allNodes.select('text')
    .text(d => {
      // Show value in forward mode
      if (state.showValues && state.mode === 'forward') {
        return d.value.data.toFixed(2);
      }
      // Show gradient in backward mode
      if (state.showGradients && state.mode === 'backward') {
        return d.value.grad.toFixed(3);
      }
      return '';
    })
    .attr('fill', d => {
      // Ensure text is readable on colored backgrounds
      const fillColor = state.mode === 'forward'
        ? scales.activationColor(sigmoid(d.value.data))
        : 'white';
      const brightness = d3.hsl(fillColor).l;
      return brightness > 0.6 ? '#1f2937' : '#ffffff';
    });

  // Add smooth transitions
  allNodes.select('circle')
    .transition()
    .duration(300)
    .attr('fill', d => {
      if (state.activeNodes.size > 0 && !state.activeNodes.has(d.id)) {
        return 'white';
      }
      if (state.mode === 'forward') {
        return scales.activationColor(sigmoid(d.value.data));
      }
      if (state.mode === 'backward' && state.showGradients) {
        return scales.gradientColor(d.value.grad);
      }
      return 'white';
    });

  nodeGroups.exit().remove();
}