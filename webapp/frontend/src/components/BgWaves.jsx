import React from "react";

/**
 * Animated concentric grey rings as a fixed background.
 * Each ring starts at a different delay so they flow continuously.
 */
const WAVES = [0, 1.6, 3.2, 4.8, 6.4];

export default function BgWaves() {
  return (
    <div className="bg-waves" aria-hidden="true">
      {WAVES.map((delay, i) => (
        <div
          key={i}
          className="bg-wave"
          style={{
            width:           `${120 + i * 60}px`,
            height:          `${120 + i * 60}px`,
            animationDelay:  `${delay}s`,
            animationDuration: "8s",
          }}
        />
      ))}
    </div>
  );
}
