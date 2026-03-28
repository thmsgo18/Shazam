import React, { useState, useRef } from "react";
import { t } from "../i18n";

const ACCEPT = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm", "audio/x-wav"];

export default function DropZone({ lang, onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  };

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onFile(file);
    e.target.value = "";
  };

  const cls = [
    "dropzone",
    dragging ? "dropzone--dragging" : "",
    disabled ? "dropzone--disabled" : "",
  ].filter(Boolean).join(" ");

  return (
    <div
      className={cls}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT.join(",")}
        style={{ display: "none" }}
        onChange={handleChange}
      />
      <UploadIcon dragging={dragging} />
      <div>
        <span className="dropzone-text">
          {dragging ? t(lang, "dropzoneActive") : t(lang, "dropzone")}
        </span>
        {!dragging && (
          <span className="dropzone-sub" style={{ marginLeft: 8 }}>
            — {t(lang, "dropzoneSub")}
          </span>
        )}
      </div>
    </div>
  );
}

function UploadIcon({ dragging }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke={dragging ? "var(--blue-bright)" : "var(--text-subtle)"}
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"
      style={{ transition: "stroke 0.2s", flexShrink: 0 }}
    >
      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );
}
