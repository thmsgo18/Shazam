import React, { useState } from "react";
import { t } from "../i18n";
import AlbumCover from "./AlbumCover";
import StreamingLinks from "./StreamingLinks";
import Recommendations from "./Recommendations";

function formatScore(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "n/a";
}

export default function ResultView({ lang, data, onReset, theme, showDebug }) {
  const [leaving, setLeaving] = useState(false);

  if (!data) return null;
  const { results, recommendations } = data;

  const handleBack = () => {
    setLeaving(true);
    setTimeout(onReset, 380);
  };

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
    <div className={`result-split${leaving ? " result-split--leaving" : ""}`}>

      {/* ── LEFT: back arrow + cover + info ── */}
      <div className="result-split-left">

        <button className="back-btn" onClick={handleBack} aria-label={t(lang, "back")}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5"/>
            <polyline points="12 19 5 12 12 5"/>
          </svg>
          <span>{t(lang, "back")}</span>
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

      {/* ── RIGHT: streaming + recommendations  OR  debug scores ── */}
      <div className="result-split-right">

        {showDebug ? (
          <div className="debug-panel">
            <p className="result-section-label debug-panel-title">
              {t(lang, "debugTitle")}
            </p>
            <ol className="debug-list">
              {results.slice(0, 10).map((r) => (
                <li key={r.track_id} className={`debug-item${r.rank === 1 ? " debug-item--best" : ""}`}>
                  <span className="debug-rank">#{r.rank}</span>
                  <span className="debug-info">
                    <span className="debug-title">{r.title}</span>
                    <span className="debug-artist">{r.artist}</span>
                  </span>
                  <span className="debug-score">
                    {`Final ${formatScore(r.score)} | FAISS ${formatScore(r.score_faiss)} | FP ${formatScore(r.score_fp)}`}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ) : (
          <>
            <div>
              <p className="result-section-label">{t(lang, "streaming")}</p>
              <StreamingLinks lang={lang} streaming={best.streaming} />
            </div>

            {recommendations?.length > 0 && (
              <div>
                <p className="result-section-label">{t(lang, "recs")}</p>
                <Recommendations lang={lang} recs={recommendations} theme={theme} />
              </div>
            )}
          </>
        )}

      </div>

    </div>
  );
}
