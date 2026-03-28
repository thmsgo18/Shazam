import React, { useState } from "react";
import RecModal from "./RecModal";

export default function Recommendations({ lang, recs }) {
  const [selected, setSelected] = useState(null);

  if (!recs?.length) return null;
  return (
    <>
      <div className="recs-grid">
        {recs.map((rec) => (
          <div
            key={rec.track_id}
            className="rec-card"
            onClick={() => setSelected(rec)}
            style={{ cursor: "pointer" }}
          >
            {/* Cover */}
            <div className="rec-cover-wrap">
              {rec.cover_url
                ? <img src={rec.cover_url} alt={rec.title} />
                : (
                  <div className="rec-cover-placeholder">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5"/>
                      <circle cx="12" cy="12" r="3" fill="rgba(255,255,255,0.12)"/>
                      <circle cx="12" cy="12" r="6" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
                    </svg>
                  </div>
                )
              }
            </div>

            {/* Title + artist */}
            <div className="rec-info">
              <p className="rec-title">{rec.title}</p>
              <p className="rec-artist">{rec.artist}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal */}
      {selected && (
        <RecModal
          lang={lang}
          rec={selected}
          onClose={() => setSelected(null)}
        />
      )}
    </>
  );
}
