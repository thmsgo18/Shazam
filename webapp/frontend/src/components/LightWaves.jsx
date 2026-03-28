import React, { useEffect, useRef } from "react";

const WAVES = [
  { speed: 0.38, freqMult: 1.0,  baseAmp: 14, strokeW: 1.2, color: "rgba(37,99,235,0.13)"  },
  { speed: 0.30, freqMult: 0.85, baseAmp: 20, strokeW: 1.5, color: "rgba(59,130,246,0.18)" },
  { speed: 0.46, freqMult: 1.2,  baseAmp: 11, strokeW: 1.0, color: "rgba(96,165,250,0.14)" },
  { speed: 0.25, freqMult: 0.7,  baseAmp: 24, strokeW: 1.8, color: "rgba(37,99,235,0.10)"  },
  { speed: 0.52, freqMult: 1.4,  baseAmp: 10, strokeW: 0.9, color: "rgba(147,197,253,0.20)"},
  { speed: 0.34, freqMult: 0.95, baseAmp: 18, strokeW: 1.3, color: "rgba(59,130,246,0.12)" },
  { speed: 0.42, freqMult: 1.15, baseAmp: 13, strokeW: 1.1, color: "rgba(29,78,216,0.09)"  },
];

const STEPS = 120;

export default function LightWaves() {
  const svgRef   = useRef(null);
  const mouseRef = useRef({ x: 0.5, y: 0.5 });
  const rafRef   = useRef(null);

  useEffect(() => {
    const onMove = (e) => {
      mouseRef.current = {
        x: e.clientX / window.innerWidth,
        y: e.clientY / window.innerHeight,
      };
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const paths = Array.from(svg.querySelectorAll("path"));

    const animate = (ts) => {
      const t  = ts * 0.001;
      const W  = window.innerWidth;
      const H  = window.innerHeight;
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;

      paths.forEach((path, i) => {
        const wave = WAVES[i];
        // Y position of this wave (evenly distributed)
        const normY  = (i + 1) / (WAVES.length + 1);
        const baseY  = normY * H;

        // How close is the mouse vertically (0=far, 1=right on the wave)
        const distY    = Math.abs(my - normY);
        const yProx    = Math.max(0, 1 - distY * 3.5);

        const freq  = (2 * Math.PI / W) * wave.freqMult * 1.8;
        const phase = t * wave.speed;

        let d = "";
        for (let s = 0; s <= STEPS; s++) {
          const x     = (s / STEPS) * W;
          const normX = x / W;

          // How close is the mouse horizontally
          const distX  = Math.abs(normX - mx);
          const xProx  = Math.max(0, 1 - distX * 2.8);

          // Amplitude: base + bump where mouse is near
          const proximity = yProx * (0.25 + xProx * 0.75);
          const amp = wave.baseAmp + proximity * 62;

          const y = baseY + amp * Math.sin(x * freq + phase);
          d += s === 0 ? `M ${x.toFixed(1)} ${y.toFixed(1)}` : ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
        }
        path.setAttribute("d", d);
      });

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  return (
    <svg ref={svgRef} className="light-waves" xmlns="http://www.w3.org/2000/svg">
      {WAVES.map((w, i) => (
        <path
          key={i}
          fill="none"
          stroke={w.color}
          strokeWidth={w.strokeW}
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}
