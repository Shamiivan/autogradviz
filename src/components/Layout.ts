import * as d3 from 'd3';
import type { NetworkData, LayoutConfig } from './types';

/**
 * Calculate x,y positions for all nodes in the network
 * Uses a layered layout with nodes centered vertically in each layer
 */
export function calculateLayout(
  networkData: NetworkData,
  config: LayoutConfig
): NetworkData {
  const { width, height, nodeSpacing } = config;
  const numLayers = networkData.layers.length;

  // Calculate x position for each layer
  // Add padding on left and right
  const layerX = d3.scaleLinear()
    .domain([0, numLayers - 1])
    .range([100, width - 100]);

  // Position each node
  networkData.nodes.forEach(node => {
    const layer = networkData.layers.find(l => l.layer === node.layer);
    if (!layer) return;

    const layerSize = layer.size;

    // Center nodes vertically within their layer
    const totalHeight = (layerSize - 1) * nodeSpacing;
    const startY = (height - totalHeight) / 2;

    node.x = layerX(node.layer);
    node.y = startY + node.indexInLayer * nodeSpacing;
  });

  return networkData;
}

/**
 * Alternative layout: Force-directed layout
 * Useful for visualizing the computational graph structure
 */
export function forceDirectedLayout(
  networkData: NetworkData,
  config: LayoutConfig
): NetworkData {
  const { width, height } = config;

  // Create force simulation
  const simulation = d3.forceSimulation(networkData.nodes as any)
    .force('link', d3.forceLink(networkData.edges as any)
      .id((d: any) => d.id)
      .distance(100)
    )
    .force('charge', d3.forceManyBody().strength(-300))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(30));

  // Run simulation synchronously for initial positions
  for (let i = 0; i < 300; i++) {
    simulation.tick();
  }

  simulation.stop();

  return networkData;
}