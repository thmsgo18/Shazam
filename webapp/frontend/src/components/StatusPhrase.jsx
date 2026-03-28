import React, { useState, useEffect } from "react";

const PHRASES = {
  recording: {
    fr: [
      "On écoute… 🎵",
      "Vibing avec toi…",
      "C'est quoi ce son ?",
      "Les oreilles grandes ouvertes…",
      "Capte bien le son…",
      "On est à l'écoute…",
      "Quelques secondes encore…",
      "Hum hum hum…",
    ],
    en: [
      "Listening… 🎵",
      "Vibing with you…",
      "What's that sound?",
      "Ears wide open…",
      "Catching the beat…",
      "On it…",
      "Just a few more seconds…",
      "Hm hm hm…",
    ],
  },
  analyzing: {
    fr: [
      "On cherche…",
      "Ça arrive…",
      "Fouille dans la base…",
      "Presque là…",
      "Comparaison en cours…",
      "Les algorithmes bossent…",
      "On y est presque…",
      "Analyse approfondie…",
    ],
    en: [
      "Searching…",
      "On it…",
      "Digging through the database…",
      "Almost there…",
      "Comparing signatures…",
      "Algorithms at work…",
      "Getting closer…",
      "Deep analysis…",
    ],
  },
};

export default function StatusPhrase({ status, lang }) {
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState(true);

  const phrases = PHRASES[status]?.[lang] ?? [];

  useEffect(() => {
    setIndex(0);
    setVisible(true);
  }, [status, lang]);

  useEffect(() => {
    if (!phrases.length) return;
    const interval = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setIndex((i) => (i + 1) % phrases.length);
        setVisible(true);
      }, 300);
    }, 2800);
    return () => clearInterval(interval);
  }, [phrases.length]);

  if (!phrases.length) return null;

  return (
    <p
      className="status-phrase"
      style={{ opacity: visible ? 1 : 0, transition: "opacity 0.3s ease" }}
    >
      {phrases[index]}
    </p>
  );
}
