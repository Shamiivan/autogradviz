import type { FC } from "react";
import type { IrisSample } from "../micrograd/preprocessor";
import { IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES } from "../utils";

interface FeatureScatterProps {
  samples: IrisSample[];
  currentSample?: IrisSample;
  width?: number;
  height?: number;
  featureX?: number;
  featureY?: number;
}

const CLASS_COLORS = ["#2563eb", "#ec4899", "#f59e0b"];

export const FeatureScatter: FC<FeatureScatterProps> = ({
  samples,
  currentSample,
  width = 240,
  height = 220,
  featureX = 0,
  featureY = 2
}) => {
  if (samples.length === 0) {
    return null;
  }

  const padding = 24;
  const svgWidth = Math.max(width - 32, 120);
  const svgHeight = Math.max(height - 32, 140);
  const innerWidth = svgWidth - padding * 2;
  const innerHeight = svgHeight - padding * 2;

  const toPoint = (sample: IrisSample) => ({
    x: sample.inputs[featureX]?.data ?? 0,
    y: sample.inputs[featureY]?.data ?? 0,
    classIndex: sample.outputs.findIndex(output => output.data === 1)
  });

  const points = samples.map(toPoint);
  const activePoint = currentSample ? toPoint(currentSample) : null;

  const xLabel = IRIS_FEATURE_NAMES[featureX] ?? `Feature ${featureX + 1}`;
  const yLabel = IRIS_FEATURE_NAMES[featureY] ?? `Feature ${featureY + 1}`;

  return (
    <div style={{
      width,
      minWidth: width,
      maxWidth: "100%",
      background: "white",
      borderRadius: "12px",
      border: "1px solid #e5e7eb",
      padding: "16px",
      boxShadow: "0 12px 24px rgba(15, 23, 42, 0.08)"
    }}>
      <div style={{ fontWeight: 600, fontSize: "13px", color: "#0f172a", marginBottom: "12px" }}>
        Feature Space ({xLabel} vs {yLabel})
      </div>
      <svg width={svgWidth} height={svgHeight}>
        <g transform={`translate(${padding}, ${padding})`}>
          {/* Axes */}
          <line
            x1={0}
            y1={innerHeight}
            x2={innerWidth}
            y2={innerHeight}
            stroke="#cbd5f5"
            strokeWidth={1}
          />
          <line
            x1={0}
            y1={0}
            x2={0}
            y2={innerHeight}
            stroke="#cbd5f5"
            strokeWidth={1}
          />

          {/* Background grid */}
          {[0.25, 0.5, 0.75].map((ratio) => (
            <line
              key={`grid-x-${ratio}`}
              x1={innerWidth * ratio}
              y1={0}
              x2={innerWidth * ratio}
              y2={innerHeight}
              stroke="#e7eafc"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
          ))}
          {[0.25, 0.5, 0.75].map((ratio) => (
            <line
              key={`grid-y-${ratio}`}
              x1={0}
              y1={innerHeight * (1 - ratio)}
              x2={innerWidth}
              y2={innerHeight * (1 - ratio)}
              stroke="#e7eafc"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
          ))}

          {/* Samples */}
          {points.map((point, idx) => {
            const x = point.x * innerWidth;
            const y = innerHeight - point.y * innerHeight;
            const color = CLASS_COLORS[point.classIndex] ?? "#6b7280";

            return (
              <circle
                key={`scatter-point-${idx}`}
                cx={x}
                cy={y}
                r={3}
                fill={color}
                opacity={0.5}
              />
            );
          })}

          {/* Active sample highlight */}
          {activePoint && (
            <g transform={`translate(${activePoint.x * innerWidth}, ${innerHeight - activePoint.y * innerHeight})`}>
              <circle r={6} fill="white" stroke="#0f172a" strokeWidth={2} />
              <circle r={4} fill={CLASS_COLORS[activePoint.classIndex] ?? "#111827"} />
            </g>
          )}

          {/* Axis labels */}
          <text
            x={innerWidth / 2}
            y={innerHeight + 24}
            textAnchor="middle"
            fontSize={11}
            fill="#475569"
          >
            {xLabel}
          </text>
          <text
            x={-30}
            y={innerHeight / 2}
            textAnchor="middle"
            fontSize={11}
            fill="#475569"
            transform={`rotate(-90, ${-30}, ${innerHeight / 2})`}
          >
            {yLabel}
          </text>
        </g>
      </svg>
      <div style={{
        marginTop: "12px",
        display: "flex",
        flexWrap: "wrap",
        gap: "12px",
        fontSize: "11px",
        color: "#475569",
        justifyContent: "center"
      }}>
        {IRIS_CLASS_NAMES.map((name, idx) => (
          <div key={`class-legend-${name}`} style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: CLASS_COLORS[idx] ?? "#6b7280"
            }}></span>
            <span>{name}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FeatureScatter;
