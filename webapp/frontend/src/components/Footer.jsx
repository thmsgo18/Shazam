import React from "react";
import { t } from "../i18n";

export default function Footer({ lang }) {
  return (
    <footer className="footer">
      <a
        href="https://github.com/thmsgo18/Shazam"
        target="_blank"
        rel="noopener noreferrer"
        className="footer-link"
      >
        {t(lang, "footer")}
      </a>
    </footer>
  );
}
