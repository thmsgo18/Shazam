import React, { useState, useEffect, useCallback } from "react";
import Header        from "./components/Header";
import BgWaves       from "./components/BgWaves";
import LightWaves    from "./components/LightWaves";
import ListenButton  from "./components/ListenButton";
import StatusPhrase  from "./components/StatusPhrase";
import DropZone      from "./components/DropZone";
import ResultView    from "./components/ResultView";
import Footer        from "./components/Footer";
import useRecorder   from "./hooks/useRecorder";
import { t }         from "./i18n";

const API_BASE       = "";
const DEFAULT_DURATION = 15;

export default function App() {
  const [lang,      setLang]      = useState("fr");
  const [theme,     setTheme]     = useState("dark");   // "dark" | "light"
  const [appConfig, setAppConfig] = useState({ listen_duration: DEFAULT_DURATION });
  const [status,    setStatus]    = useState("idle");
  const [result,    setResult]    = useState(null);
  const [errorMsg,  setErrorMsg]  = useState("");
  const [showDebug, setShowDebug] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((r) => r.json())
      .then((cfg) => setAppConfig(cfg))
      .catch(() => {});
  }, []);

  const duration = appConfig.listen_duration ?? DEFAULT_DURATION;
  const { isRecording, countdown, startRecording, stopRecording } = useRecorder(duration);

  useEffect(() => { if (isRecording) setStatus("recording"); }, [isRecording]);

  const identify = useCallback(async (blob, filename = "recording.webm") => {
    setStatus("analyzing");
    setResult(null);
    setErrorMsg("");
    const form = new FormData();
    form.append("file", blob, filename);
    try {
      const res = await fetch(`${API_BASE}/api/identify`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Server error");
      }
      setResult(await res.json());
      setStatus("result");
    } catch (err) {
      setErrorMsg(err.message);
      setStatus("error");
    }
  }, []);

  const handleListenClick = useCallback(() => {
    if (status === "recording") { stopRecording(); return; }
    setStatus("recording");
    startRecording((blob) => identify(blob, "recording.webm"));
  }, [status, startRecording, stopRecording, identify]);

  const handleFile = useCallback((file) => identify(file, file.name), [identify]);

  const reset = useCallback(() => {
    setStatus("idle");
    setResult(null);
    setErrorMsg("");
    setShowDebug(false);
  }, []);

  const isIdle   = status === "idle";
  const isActive = status === "recording" || status === "analyzing";
  const isResult = status === "result";
  const isError  = status === "error";

  const coverUrl   = result?.results?.[0]?.cover_url || null;
  const albumMode  = isResult; // fond pochette toujours actif en vue résultat
  const lightAlbum = theme === "light" && isResult;

  return (
    <div className={`app theme-${theme}${lightAlbum ? " theme-light-result" : ""}${albumMode ? " theme-album-result" : ""}`}>

      {theme === "dark"  && !isResult && <LightWaves variant="dark" />}
      {theme === "light" && !isResult && <LightWaves variant="light" />}

      {/* Fond pochette flouté dans la vue résultat */}
      {albumMode && coverUrl && (
        <div className={`album-bg${theme === "dark" ? " album-bg--dark" : ""}`}
             style={{ backgroundImage: `url(${coverUrl})` }} />
      )}

      <Header
        lang={lang}
        onToggleLang={() => setLang((l) => l === "fr" ? "en" : "fr")}
        theme={theme}
        onToggleTheme={() => setTheme((t) => t === "dark" ? "light" : "dark")}
        showDebug={showDebug}
        onToggleDebug={isResult ? () => setShowDebug((v) => !v) : undefined}
      />

      <main className={`main${isResult ? " main--result" : ""}`}>

        {isIdle && (
          <div className="hero">
            <p className="hero-title">{t(lang, "tagline")}</p>
            <ListenButton
              lang={lang}
              isRecording={false}
              isAnalyzing={false}
              countdown={duration}
              duration={duration}
              onClick={handleListenClick}
            />
            <div className="dropzone-wrap">
              <div className="divider">
                <hr /><span>{t(lang, "dropzoneOr")}</span><hr />
              </div>
              <DropZone lang={lang} onFile={handleFile} disabled={false} />
            </div>
          </div>
        )}

        {isActive && (
          <div className="hero">
            <ListenButton
              lang={lang}
              isRecording={status === "recording"}
              isAnalyzing={status === "analyzing"}
              countdown={countdown}
              duration={duration}
              onClick={status === "recording" ? handleListenClick : undefined}
            />
            <StatusPhrase status={status} lang={lang} />
          </div>
        )}

        {isError && (
          <div className="error-box">
            <p className="error-title">⚠ {t(lang, "error")}</p>
            {errorMsg && <p className="error-detail">{errorMsg}</p>}
            <button className="btn-reset" onClick={reset}>{t(lang, "tryAgain")}</button>
          </div>
        )}

        {isResult && (
          <ResultView lang={lang} data={result} onReset={reset} theme={theme} showDebug={showDebug} />
        )}

      </main>

      <Footer />
    </div>
  );
}
