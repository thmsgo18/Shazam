import { useRef, useState, useCallback } from "react";

/**
 * useRecorder(durationSec)
 * Records from the microphone for `durationSec` seconds using MediaRecorder.
 * Returns { isRecording, countdown, startRecording, blob }
 */
export default function useRecorder(durationSec = 15, microphoneDeniedMessage = "Microphone access denied.") {
  const [isRecording, setIsRecording]   = useState(false);
  const [countdown,   setCountdown]     = useState(durationSec);
  const [blob,        setBlob]          = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const timerRef         = useRef(null);

  const startRecording = useCallback(async (onDone) => {
    setBlob(null);
    chunksRef.current = [];
    setCountdown(durationSec);

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      alert(microphoneDeniedMessage);
      return;
    }

    // Pick the best available MIME type
    const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/ogg"]
      .find((m) => MediaRecorder.isTypeSupported(m)) ?? "";

    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    mediaRecorderRef.current = recorder;

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const audioBlob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
      setBlob(audioBlob);
      setIsRecording(false);
      if (onDone) onDone(audioBlob);
    };

    recorder.start(250); // collect data every 250 ms
    setIsRecording(true);

    // Countdown ticker
    let remaining = durationSec;
    timerRef.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0) {
        clearInterval(timerRef.current);
        if (recorder.state !== "inactive") recorder.stop();
      }
    }, 1000);
  }, [durationSec, microphoneDeniedMessage]);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    if (mediaRecorderRef.current?.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  return { isRecording, countdown, blob, startRecording, stopRecording };
}
