import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { MLP } from '../micrograd/nn';
import { Value } from '../micrograd/engine';
import { extractNetworkData, updateNodeValues } from './ExtractData';
import { calculateLayout } from "./Layout"
import { renderNetwork } from "./Render"
import {
  AnimationController,
  createForwardPassSteps,
  createBackwardPassSteps
} from "./Animation";
import type { NetworkData, VizState, AnimationStep, LayoutConfig } from './types';

interface NetworkVisualizerProps {
  mlp: MLP;
  inputs: Value[];
  width?: number;
  height?: number;
  onStepChange?: (step: AnimationStep) => void;
}

export function NetworkVisualizer({
  mlp,
  inputs,
  width = 800,
  height = 600,
  onStepChange
}: NetworkVisualizerProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const [controller] = useState(() => new AnimationController(
    (step, index, total) => {
      setCurrentStep(index);
      setTotalSteps(total);
      updateVisualizationState(step);
      onStepChange?.(step);
    }
  ));

  const [vizState, setVizState] = useState<VizState>({
    mode: 'architecture',
    showValues: true,
    showGradients: false,
    activeNodes: new Set(),
    activeEdges: new Set()
  });

  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Initialize network data and layout
  useEffect(() => {
    const data = extractNetworkData(mlp, inputs);
    const layoutConfig: LayoutConfig = {
      width,
      height,
      nodeRadius: 20,
      layerSpacing: width / (data.layers.length + 1),
      nodeSpacing: 60
    };
    const layoutData = calculateLayout(data, layoutConfig);
    setNetworkData(layoutData);
  }, [mlp, inputs, width, height]);

  // Render visualization whenever state changes
  useEffect(() => {
    if (!svgRef.current || !networkData) return;

    const svg = d3.select(svgRef.current);
    renderNetwork(svg, networkData, vizState);
  }, [networkData, vizState]);

  // Update visualization state based on animation step
  const updateVisualizationState = (step: AnimationStep) => {
    setVizState(prev => ({
      ...prev,
      mode: step.type === 'forward' ? 'forward' : 'backward',
      showGradients: step.type === 'backward',
      activeNodes: new Set(step.activeNodes),
      activeEdges: new Set(step.activeEdges)
    }));
  };

  // Control handlers
  const handlePlayForward = () => {
    if (!networkData) return;

    // Compute forward pass and update node values
    updateNodeValues(networkData, mlp, inputs);

    // Create animation steps
    const steps = createForwardPassSteps(mlp, inputs, networkData);
    controller.setSteps(steps);
    controller.reset();
    controller.play(500);
    setIsPlaying(true);
  };

  const handlePlayBackward = () => {
    if (!networkData) return;

    // First do forward pass
    updateNodeValues(networkData, mlp, inputs);
    const output = mlp.forward(inputs);

    // Do backward pass
    mlp.zeroGrad();
    if (Array.isArray(output)) {
      output[0].backward();
    } else {
      output.backward();
    }

    // Create animation steps
    const steps = createBackwardPassSteps(mlp, networkData);
    controller.setSteps(steps);
    controller.reset();
    controller.play(500);
    setIsPlaying(true);
  };

  const handlePause = () => {
    controller.pause();
    setIsPlaying(false);
  };

  const handleStepForward = () => {
    controller.stepForward();
  };

  const handleStepBackward = () => {
    controller.stepBackward();
  };

  const handleReset = () => {
    controller.reset();
    setIsPlaying(false);
    setVizState(prev => ({
      ...prev,
      mode: 'architecture',
      activeNodes: new Set(),
      activeEdges: new Set()
    }));
  };

  const toggleShowValues = () => {
    setVizState(prev => ({ ...prev, showValues: !prev.showValues }));
  };

  const toggleShowGradients = () => {
    setVizState(prev => ({ ...prev, showGradients: !prev.showGradients }));
  };

  return (
    <div className="network-visualizer">
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

      <div className="controls" style={{
        marginTop: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {/* Main controls */}
        <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
          <button
            onClick={handleReset}
            disabled={currentStep === 0}
            style={buttonStyle}
          >
            ⟲ Reset
          </button>

          <button
            onClick={handleStepBackward}
            disabled={controller.isAtStart()}
            style={buttonStyle}
          >
            ← Step Back
          </button>

          <button
            onClick={isPlaying ? handlePause : handlePlayForward}
            style={{ ...buttonStyle, fontWeight: 'bold' }}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play Forward'}
          </button>

          <button
            onClick={handleStepForward}
            disabled={controller.isAtEnd()}
            style={buttonStyle}
          >
            Step Forward →
          </button>

          <button
            onClick={handlePlayBackward}
            style={buttonStyle}
          >
            ⏮ Play Backward
          </button>
        </div>

        {/* Display options */}
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <input
              type="checkbox"
              checked={vizState.showValues}
              onChange={toggleShowValues}
            />
            Show Values
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <input
              type="checkbox"
              checked={vizState.showGradients}
              onChange={toggleShowGradients}
              disabled={vizState.mode !== 'backward'}
            />
            Show Gradients
          </label>
        </div>

        {/* Progress indicator */}
        <div style={{ textAlign: 'center', fontSize: '14px', color: '#6b7280' }}>
          Step {currentStep + 1} of {totalSteps}
          {vizState.mode !== 'architecture' && (
            <span style={{ marginLeft: '12px', fontWeight: '500' }}>
              Mode: {vizState.mode === 'forward' ? 'Forward Pass' : 'Backward Pass'}
            </span>
          )}
        </div>

        {/* Progress bar */}
        <div style={{
          width: '100%',
          height: '4px',
          background: '#e5e7eb',
          borderRadius: '2px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${controller.getProgress() * 100}%`,
            height: '100%',
            background: vizState.mode === 'forward' ? '#3b82f6' : '#f59e0b',
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>
    </div>
  );
}

const buttonStyle: React.CSSProperties = {
  padding: '8px 16px',
  border: '1px solid #d1d5db',
  borderRadius: '6px',
  background: 'white',
  cursor: 'pointer',
  fontSize: '14px',
  transition: 'all 0.2s',
  // add text color 
  color: '#374151',
};