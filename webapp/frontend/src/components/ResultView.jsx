import React from "react";
import { t } from "../i18n";
import AlbumCover from "./AlbumCover";
import StreamingLinks from "./StreamingLinks";
import Recommendations from "./Recommendations";

export default function ResultView({ lang, data, onReset }) {
  if (!data) return null;
  const { results, recommendations } = data;

  if (!results?.length) {
    return (
      <div className="error-box">
        <p style={{ color: "var(--text-muted)", marginBottom: 20 }}>{t(lang, "noResult")}</p>
        <button className="btn-reset" onClick={onReset}>{t(lang, "tryAgain")}</button>
      </div>
    );
  }

  const best = results[0];

  return (
    <div className="result-split">

      {/* ── LEFT: back arrow + cover + info ── */}
      <div className="result-split-left">

        <button className="back-btn" onClick={onReset} aria-label="Retour">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5"/>
            <polyline points="12 19 5 12 12 5"/>
          </svg>
          <span>{lang === "fr" ? "Retour" : "Back"}</span>
        </button>

        <div className="result-left-inner">
          <AlbumCover url={best.cover_url} title={best.title} fluid radius={16} />
          <div className="result-track-info">
            <h2 className="result-title">{best.title}</h2>
            {best.album && <p className="result-album">{best.album}</p>}
            <p className="result-artist">{best.artist}</p>
          </div>
        </div>

      </div>

      {/* ── RIGHT: streaming + recommendations ── */}
      <div className="result-split-right">

        <div>
          <p className="result-section-label">{t(lang, "streaming")}</p>
          <StreamingLinks lang={lang} streaming={best.streaming} />
        </div>

        {recommendations?.length > 0 && (
          <div>
            <p className="result-section-label">{t(lang, "recs")}</p>
            <Recommendations lang={lang} recs={recommendations} />
          </div>
        )}

      </div>

    </div>
  );
}
