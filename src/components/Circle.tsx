import { useEffect, useRef } from "react";
import * as d3 from "d3";

interface CircleProps {
  x?: number;
  y?: number;
  radius: number;
  color: string;
  numberOfCircles?: number;
}

export default function Circle({ x = 0, y = 0, radius, color, numberOfCircles = 1 }: CircleProps) {
  const ref = useRef<SVGSVGElement | null>(null);
  // Define nodes for each layer
  const inputNodes = [
    { x: 100, y: 50 },
    { x: 100, y: 120 },
    { x: 100, y: 190 },
    { x: 100, y: 260 }
  ];

  const hiddenNodes = [
    { x: 250, y: 30 },
    { x: 250, y: 100 },
    { x: 250, y: 170 },
    { x: 250, y: 240 },
    { x: 250, y: 310 }
  ];

  const outputNodes = [
    { x: 400, y: 85 },
    { x: 400, y: 170 },
    { x: 400, y: 255 }
  ];

  useEffect(() => {
    if (!ref.current) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove(); // Clear previous drawings
    // Connect inputs to hidden (4 × 5 = 20 lines)
    inputNodes.forEach(input => {
      hiddenNodes.forEach(hidden => {
        svg.append('line')
          .attr('x1', input.x)
          .attr('y1', input.y)
          .attr('x2', hidden.x)
          .attr('y2', hidden.y)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', 1);
      });
    });

    // Connect hidden to output (5 × 3 = 15 lines)
    hiddenNodes.forEach(hidden => {
      outputNodes.forEach(output => {
        svg.append('line')
          .attr('x1', hidden.x)
          .attr('y1', hidden.y)
          .attr('x2', output.x)
          .attr('y2', output.y)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', 1);
      });
    });
    // Draw input nodes
    inputNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', 'steelblue');
    });

    // Draw hidden nodes
    hiddenNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', 'orange');
    });

    // Draw input nodes (light blue)
    inputNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#87CEEB');  // Light blue
    });

    // Draw hidden nodes (medium blue)
    hiddenNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#4682B4');  // Steel blue (medium)
    });

    // Draw output nodes (dark blue)
    outputNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#000080');  // Navy blue (dark)
    });

    // Add layer labels
    svg.append('text')
      .attr('x', 100)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Input Layer');

    svg.append('text')
      .attr('x', 250)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Hidden Layer');

    svg.append('text')
      .attr('x', 400)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Output Layer');

    // Draw input nodes (light blue)
    inputNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#87CEEB');  // Light blue
    });

    // Draw hidden nodes (medium blue)
    hiddenNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#4682B4');  // Steel blue (medium)
    });

    // Draw output nodes (dark blue)
    outputNodes.forEach(node => {
      svg.append('circle')
        .attr('cx', node.x)
        .attr('cy', node.y)
        .attr('r', 15)
        .attr('fill', '#000080');  // Navy blue (dark)
    });
  }, [x, y, radius, color, numberOfCircles]);

  return <svg ref={ref} width={x * numberOfCircles * 10} height={y * 10}></svg>;
}
