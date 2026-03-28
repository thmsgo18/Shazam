import React from "react";
import { t } from "../i18n";

export default function ListenButton({
  lang,
  isRecording,
  isAnalyzing,
  countdown,
  duration,
  onClick,
}) {
  const R        = 118;
  const C        = 2 * Math.PI * R;
  const progress = isRecording ? ((duration - countdown) / duration) * 100 : 0;

  const btnClass = [
    "listen-btn",
    isAnalyzing  ? "listen-btn--analyzing"  :
    isRecording  ? "listen-btn--recording"  :
                   "listen-btn--idle",
  ].join(" ");

  return (
    <div className="listen-area">
      {/* Ripple rings — only when recording or analyzing */}
      {(isRecording || isAnalyzing) && (
        <>
          <span className="listen-ripple" style={{ animationDelay: "0s"    }} />
          <span className="listen-ripple" style={{ animationDelay: "0.55s" }} />
          <span className="listen-ripple" style={{ animationDelay: "1.1s"  }} />
        </>
      )}

      {/* SVG progress arc */}
      <svg className="listen-svg" viewBox="0 0 260 260">
        <circle
          cx="130" cy="130" r={R}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="2.5"
        />
        {isRecording && (
          <circle
            cx="130" cy="130" r={R}
            fill="none"
            stroke="rgba(59,130,246,0.8)"
            strokeWidth="2.5"
            strokeDasharray={C}
            strokeDashoffset={C * (1 - progress / 100)}
            strokeLinecap="round"
            transform="rotate(-90 130 130)"
            style={{ transition: "stroke-dashoffset 0.9s linear" }}
          />
        )}
      </svg>

      <button
        className={btnClass}
        onClick={!isAnalyzing ? onClick : undefined}
        disabled={isAnalyzing}
        aria-label={t(lang, "listenBtn")}
      >
        {isAnalyzing ? (
          <div className="btn-spinner" />
        ) : isRecording ? (
          <>
            <MicIcon size={58} />
            <span className="listen-btn-countdown">{countdown}</span>
          </>
        ) : (
          <>
            <MicIcon size={58} />
            <span className="listen-btn-label">{t(lang, "listenBtn")}</span>
          </>
        )}
      </button>
    </div>
  );
}

function MicIcon({ size = 40 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect x="9" y="2" width="6" height="11" rx="3" fill="rgba(255,255,255,0.95)" />
      <path d="M5 10a7 7 0 0014 0"
        stroke="rgba(255,255,255,0.95)" strokeWidth="2" strokeLinecap="round"/>
      <line x1="12" y1="17" x2="12" y2="22"
        stroke="rgba(255,255,255,0.95)" strokeWidth="2" strokeLinecap="round"/>
      <line x1="8"  y1="22" x2="16" y2="22"
        stroke="rgba(255,255,255,0.95)" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}
