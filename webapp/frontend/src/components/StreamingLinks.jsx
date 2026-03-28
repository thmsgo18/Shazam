import React from "react";

const PLATFORMS = [
  { key: "youtube", label: "YouTube",     color: "#ff0000" },
  { key: "spotify", label: "Spotify",     color: "#1db954" },
  { key: "deezer",  label: "Deezer",      color: "#a238ff" },
  { key: "apple",   label: "Apple Music", color: "#fc3c44" },
];

const ICONS = {
  youtube: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M23 7s-.3-1.9-1.1-2.7c-1.1-1.1-2.3-1.1-2.8-1.2C16.2 3 12 3 12 3s-4.2 0-7.1.1c-.6.1-1.8.1-2.8 1.2C1.3 5.1 1 7 1 7S.7 9.1.7 11.3v2c0 2.1.3 4.3.3 4.3s.3 1.9 1.1 2.7c1.1 1.1 2.5 1 3.1 1.1C7.2 21.6 12 21.7 12 21.7s4.2 0 7.1-.2c.6-.1 1.8-.1 2.8-1.2.8-.8 1.1-2.7 1.1-2.7s.3-2.1.3-4.3v-2C23.3 9.1 23 7 23 7zM9.7 15.5V8.4l7.6 3.6-7.6 3.5z"/>
    </svg>
  ),
  spotify: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4.64 14.36a.624.624 0 01-.86.21c-2.36-1.44-5.33-1.77-8.83-.97a.624.624 0 01-.28-1.22c3.83-.87 7.11-.5 9.76 1.12.3.18.39.57.21.86zm1.24-2.75a.78.78 0 01-1.07.26c-2.7-1.66-6.81-2.14-10-.17a.78.78 0 01-1.07-.27.78.78 0 01.27-1.07c3.65-2.21 8.18-1.67 11.32.2.37.23.48.71.55 1.05zm.1-2.82c-3.23-1.92-8.56-2.1-11.64-.97a.938.938 0 01-.62-1.77c3.53-1.24 9.39-1 13.07 1.13a.938.938 0 01-.81 1.61z"/>
    </svg>
  ),
  deezer: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <rect x="2" y="16" width="3" height="3" rx="0.5"/>
      <rect x="7" y="13" width="3" height="6" rx="0.5"/>
      <rect x="12" y="9" width="3" height="10" rx="0.5"/>
      <rect x="17" y="5" width="3" height="14" rx="0.5"/>
    </svg>
  ),
  apple: (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.54 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701z"/>
    </svg>
  ),
};

export default function StreamingLinks({ lang, streaming }) {
  if (!streaming) return null;
  return (
    <div className="streaming-links">
      {PLATFORMS.map(({ key, label, color }) =>
        streaming[key] ? (
          <a
            key={key}
            href={streaming[key]}
            target="_blank"
            rel="noopener noreferrer"
            className="streaming-chip"
            style={{ "--chip-color": color }}
          >
            {ICONS[key]}
            {label}
          </a>
        ) : null
      )}
    </div>
  );
}
