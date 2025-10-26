import type { FC, CSSProperties } from "react";

interface LegendBadgeProps {
  label: string;
  background: string;
  border: string;
  size?: number;
}

interface LegendOverlayProps {
  floating?: boolean;
}

const Badge: FC<LegendBadgeProps> = ({ label, background, border, size = 16 }) => (
  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
    <div
      style={{
        width: size,
        height: size,
        borderRadius: "50%",
        background,
        border: `2px solid ${border}`
      }}
    ></div>
    <span>{label}</span>
  </div>
);

const ForwardLegend = () => (
  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
    <div style={{ fontWeight: 700, fontSize: "12px", color: "#1d4ed8" }}>Forward Pass (activations)</div>
    <Badge label="Inactive (< 0.3)" background="white" border="#9ca3af" />
    <Badge label="Emerging (0.3 – 0.5)" background="#ef4444" border="#991b1b" />
    <Badge label="Strong (0.5 – 0.7)" background="#eab308" border="#a16207" />
    <Badge label="Dominant (> 0.7)" background="#10b981" border="#047857" size={18} />
  </div>
);

const BackwardLegend = () => (
  <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
    <div style={{ fontWeight: 700, fontSize: "12px", color: "#f97316" }}>Backward Pass (gradients)</div>
    <Badge label="Tiny (< 0.01)" background="white" border="#9ca3af" />
    <Badge label="Moderate (0.01 – 0.5)" background="#f9a8d4" border="#be185d" />
    <Badge label="Large (0.5 – 1.0)" background="#ec4899" border="#be185d" />
    <Badge label="Dominant (> 1.0)" background="#9333ea" border="#6b21a8" size={14} />
  </div>
);

export const LegendOverlay: FC<LegendOverlayProps> = ({ floating = false }) => {
  const containerStyle: CSSProperties = floating
    ? {
      position: "absolute",
      top: "16px",
      right: "16px",
      background: "rgba(255, 255, 255, 0.95)",
      border: "1px solid #e5e7eb",
      borderRadius: "12px",
      padding: "14px 16px",
      width: "220px",
      fontSize: "12px",
      color: "#374151",
      boxShadow: "0 12px 24px rgba(15, 23, 42, 0.12)",
      backdropFilter: "blur(8px)",
      pointerEvents: "none",
      display: "flex",
      flexDirection: "column",
      gap: "12px"
    }
    : {
      marginTop: "24px",
      background: "white",
      border: "1px solid #e5e7eb",
      borderRadius: "12px",
      padding: "16px 20px",
      fontSize: "12px",
      color: "#374151",
      display: "flex",
      flexWrap: "wrap",
      gap: "24px",
      justifyContent: "center",
      boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)"
    };

  return (
    <div style={containerStyle}>
      <ForwardLegend />
      <BackwardLegend />
    </div>
  );
};

export default LegendOverlay;
