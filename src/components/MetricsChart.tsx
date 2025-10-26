import type { FC } from "react";

export interface EpochMetric {
  epoch: number;
  avgLoss: number;
  accuracy: number; // expressed as 0-1
}

interface MetricsChartProps {
  metrics: EpochMetric[];
  width?: number;
  height?: number;
}

const DEFAULT_WIDTH = 860;
const DEFAULT_HEIGHT = 260;
const MARGIN = { top: 24, right: 48, bottom: 36, left: 56 };

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

export const MetricsChart: FC<MetricsChartProps> = ({
  metrics,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT
}) => {
  if (metrics.length === 0) {
    return null;
  }

  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const maxLoss = Math.max(...metrics.map(m => m.avgLoss), 0.0001);
  const maxAccuracy = 1; // accuracy already in 0-1 range

  const xStep = metrics.length > 1 ? innerWidth / (metrics.length - 1) : 0;
  const xForIndex = (index: number) => MARGIN.left + xStep * index;
  const yForLoss = (loss: number) => MARGIN.top + innerHeight - (loss / maxLoss) * innerHeight;
  const yForAccuracy = (acc: number) => MARGIN.top + innerHeight - (acc / maxAccuracy) * innerHeight;

  const lossPath = metrics.reduce((acc, metric, index) => {
    const x = xForIndex(index);
    const y = yForLoss(metric.avgLoss);
    return acc + `${index === 0 ? "M" : "L"}${x},${y}`;
  }, "");

  const accuracyPath = metrics.reduce((acc, metric, index) => {
    const x = xForIndex(index);
    const y = yForAccuracy(clamp(metric.accuracy, 0, 1));
    return acc + `${index === 0 ? "M" : "L"}${x},${y}`;
  }, "");

  const baselineY = MARGIN.top + innerHeight;

  return (
    <svg
      width={width}
      height={height}
      role="img"
      aria-label="Loss and accuracy over epochs"
      style={{ width: "100%", height: "auto" }}
    >
      {/* Background */}
      <rect
        x={0}
        y={0}
        width={width}
        height={height}
        fill="white"
        rx={12}
      />

      {/* Grid lines */}
      {[0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = MARGIN.top + innerHeight - innerHeight * ratio;
        return (
          <line
            key={`grid-${ratio}`}
            x1={MARGIN.left}
            y1={y}
            x2={width - MARGIN.right}
            y2={y}
            stroke="#e5e7eb"
            strokeDasharray="4 4"
          />
        );
      })}

      {/* Axes */}
      <line
        x1={MARGIN.left}
        y1={baselineY}
        x2={width - MARGIN.right}
        y2={baselineY}
        stroke="#9ca3af"
        strokeWidth={1.5}
      />
      <line
        x1={MARGIN.left}
        y1={MARGIN.top}
        x2={MARGIN.left}
        y2={baselineY}
        stroke="#9ca3af"
        strokeWidth={1.5}
      />

      {/* Loss line */}
      <path
        d={lossPath}
        fill="none"
        stroke="#ef4444"
        strokeWidth={2.5}
      />

      {/* Accuracy line */}
      <path
        d={accuracyPath}
        fill="none"
        stroke="#10b981"
        strokeWidth={2.5}
      />

      {/* Loss points */}
      {metrics.map((metric, index) => (
        <circle
          key={`loss-point-${metric.epoch}`}
          cx={xForIndex(index)}
          cy={yForLoss(metric.avgLoss)}
          r={5}
          fill="white"
          stroke="#ef4444"
          strokeWidth={2}
        />
      ))}

      {/* Accuracy points */}
      {metrics.map((metric, index) => (
        <circle
          key={`accuracy-point-${metric.epoch}`}
          cx={xForIndex(index)}
          cy={yForAccuracy(clamp(metric.accuracy, 0, 1))}
          r={5}
          fill="white"
          stroke="#10b981"
          strokeWidth={2}
        />
      ))}

      {/* Epoch labels */}
      {metrics.map((metric, index) => (
        <text
          key={`epoch-label-${metric.epoch}`}
          x={xForIndex(index)}
          y={baselineY + 20}
          textAnchor="middle"
          fontSize={12}
          fill="#4b5563"
        >
          {metric.epoch + 1}
        </text>
      ))}

      {/* Left axis label (loss) */}
      <text
        x={MARGIN.left - 40}
        y={MARGIN.top - 8}
        fontSize={12}
        fill="#111827"
      >
        Loss
      </text>

      {/* Right axis label (accuracy) */}
      <text
        x={width - MARGIN.right + 8}
        y={MARGIN.top - 8}
        fontSize={12}
        fill="#111827"
        textAnchor="start"
      >
        Accuracy (%)
      </text>

      {/* Axis tick labels */}
      {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const y = MARGIN.top + innerHeight - innerHeight * ratio;
        const lossValue = (maxLoss * ratio).toFixed(2);
        const accuracyValue = (100 * ratio).toFixed(0);
        return (
          <g key={`tick-${ratio}`}>
            <text
              x={MARGIN.left - 12}
              y={y + 4}
              textAnchor="end"
              fontSize={11}
              fill="#6b7280"
            >
              {lossValue}
            </text>
            <text
              x={width - MARGIN.right + 12}
              y={y + 4}
              textAnchor="start"
              fontSize={11}
              fill="#6b7280"
            >
              {accuracyValue}
            </text>
          </g>
        );
      })}

      {/* Legend */}
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top - 16})`}>
        <rect x={0} y={-12} width={12} height={12} fill="#ef4444" rx={2} />
        <text x={18} y={-2} fontSize={12} fill="#111827">
          Avg Loss
        </text>

        <rect x={96} y={-12} width={12} height={12} fill="#10b981" rx={2} />
        <text x={112} y={-2} fontSize={12} fill="#111827">
          Accuracy
        </text>
      </g>
    </svg>
  );
};

export default MetricsChart;
