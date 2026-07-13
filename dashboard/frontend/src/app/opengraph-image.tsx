import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "MARTA — Multi-Agent Regime-Aware Trading";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#0a0a0a",
          fontFamily: "monospace",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          <div style={{ width: 20, height: 20, borderRadius: "50%", background: "#00ff7f", boxShadow: "0 0 40px #00ff7f" }} />
          <div style={{ fontSize: 120, fontWeight: 700, color: "#00ff7f", letterSpacing: -4 }}>MARTA</div>
        </div>
        <div style={{ fontSize: 30, color: "#666", marginTop: 24, letterSpacing: 2 }}>
          Multi-Agent Regime-Aware Trading Architecture
        </div>
      </div>
    ),
    { ...size },
  );
}
