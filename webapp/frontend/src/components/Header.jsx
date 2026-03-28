import React from "react";

const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
    <circle cx="12" cy="12" r="4"/>
    <line x1="12" y1="2"  x2="12" y2="5"/>
    <line x1="12" y1="19" x2="12" y2="22"/>
    <line x1="4.22" y1="4.22"  x2="6.34" y2="6.34"/>
    <line x1="17.66" y1="17.66" x2="19.78" y2="19.78"/>
    <line x1="2"  y1="12" x2="5"  y2="12"/>
    <line x1="19" y1="12" x2="22" y2="12"/>
    <line x1="4.22" y1="19.78" x2="6.34" y2="17.66"/>
    <line x1="17.66" y1="6.34" x2="19.78" y2="4.22"/>
  </svg>
);

const MoonIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

const CodeIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 18 22 12 16 6"/>
    <polyline points="8 6 2 12 8 18"/>
  </svg>
);

export default function Header({ lang, onToggleLang, theme, onToggleTheme, showDebug, onToggleDebug }) {
  return (
    <header className="header">
      <div className="header-logo">
        <svg width="34" height="34" viewBox="0 0 64 64" fill="none">
          <circle cx="32" cy="32" r="30" fill="#1a1a2e" stroke="#6c63ff" strokeWidth="2"/>
          <circle cx="32" cy="32" r="8" fill="#6c63ff"/>
          <circle cx="32" cy="32" r="16" stroke="#6c63ff" strokeWidth="1.5" opacity="0.5"/>
          <circle cx="32" cy="32" r="24" stroke="#6c63ff" strokeWidth="1" opacity="0.25"/>
        </svg>
        <span className="header-brand">Shazam</span>
      </div>

      <div className="header-actions">

        {/* Debug / code toggle — visible only when a result is shown */}
        {onToggleDebug && (
          <button
            className={`debug-btn${showDebug ? " debug-btn--active" : ""}`}
            onClick={onToggleDebug}
            aria-label="Afficher les scores de similarité"
            title={showDebug ? "Masquer les scores" : "Afficher les scores"}
          >
            <CodeIcon />
          </button>
        )}

        {/* Theme toggle: lune / soleil */}
        <button className="theme-btn" onClick={onToggleTheme} aria-label="Changer le thème">
          <span className={theme === "dark" ? "theme-btn-current" : "theme-btn-other"}>
            <MoonIcon />
          </span>
          <span className="theme-btn-sep">/</span>
          <span className={theme === "light" ? "theme-btn-current" : "theme-btn-other"}>
            <SunIcon />
          </span>
        </button>

        {/* Language toggle */}
        <button className="lang-btn" onClick={onToggleLang} aria-label="Toggle language">
          <span className="lang-btn-current">{lang === "fr" ? "FR" : "EN"}</span>
          <span className="lang-btn-sep">/</span>
          <span className="lang-btn-other">{lang === "fr" ? "EN" : "FR"}</span>
        </button>

      </div>
    </header>
  );
}
