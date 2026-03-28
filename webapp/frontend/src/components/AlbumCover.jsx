import React, { useState } from "react";

export default function AlbumCover({ url, title, size = 300, radius = 16, fluid = false }) {
  const [failed, setFailed] = useState(false);

  const style = fluid
    ? { borderRadius: radius }
    : { width: size, height: size, borderRadius: radius, flexShrink: 0 };

  if (!url || failed) {
    return (
      <div style={{
        ...style,
        background:     "var(--surface2)",
        border:         "1px solid var(--border2)",
        display:        "flex",
        alignItems:     "center",
        justifyContent: "center",
      }}>
        <svg width="30%" height="30%" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.12)" strokeWidth="1.5"/>
          <circle cx="12" cy="12" r="3"  fill="rgba(255,255,255,0.18)"/>
          <circle cx="12" cy="12" r="6"  stroke="rgba(255,255,255,0.1)" strokeWidth="1"/>
        </svg>
      </div>
    );
  }

  return (
    <img
      src={url}
      alt={title}
      style={{
        ...style,
        objectFit:  "cover",
        display:    "block",
        boxShadow:  fluid
          ? "0 32px 80px rgba(0,0,0,0.7)"
          : size > 100 ? "0 24px 60px rgba(0,0,0,0.6)" : "none",
      }}
      onError={() => setFailed(true)}
    />
  );
}
