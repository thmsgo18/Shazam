# Shazam Maison — Interface Web

Interface web React + FastAPI pour identifier des morceaux audio en temps réel.

```
webapp/
├── backend/
│   ├── server.py           # API FastAPI
│   └── requirements.txt    # fastapi, uvicorn, python-multipart
└── frontend/
    ├── package.json        # React 18 + Vite
    ├── vite.config.js      # proxy /api → localhost:8000
    ├── index.html
    └── src/
        ├── App.jsx
        ├── index.css
        ├── i18n.js
        ├── hooks/
        │   └── useRecorder.js
        └── components/
            ├── Header.jsx
            ├── ListenButton.jsx
            ├── DropZone.jsx
            ├── ResultView.jsx
            ├── StreamingLinks.jsx
            ├── Recommendations.jsx
            └── Footer.jsx
```

## Lancement en développement

### 1. Backend

```bash
# Depuis la racine du projet
pip install -r webapp/backend/requirements.txt

cd webapp/backend
uvicorn server:app --reload --port 8000
# → http://localhost:8000
```

### 2. Frontend

```bash
cd webapp/frontend
npm install
npm run dev
# → http://localhost:5173
```

> Vite proxifie automatiquement `/api/*` vers `localhost:8000`.

## Build production

```bash
cd webapp/frontend
npm run build          # génère webapp/frontend/dist/

cd ../backend
uvicorn server:app --port 8000
# Le backend sert le frontend compilé sur http://localhost:8000
```

## API

| Méthode | Route          | Description                              |
|---------|----------------|------------------------------------------|
| `GET`   | `/api/health`  | Vérification de vie                      |
| `GET`   | `/api/config`  | Retourne `listen_duration`, `confidence_ratio`, `embedding_method` |
| `POST`  | `/api/identify`| Identifie un fichier audio (multipart)   |

### Réponse `/api/identify`

```json
{
  "results": [
    {
      "rank": 1,
      "track_id": "...",
      "title": "Blinding Lights",
      "artist": "The Weeknd",
      "genre": "Pop",
      "score": 42.18,
      "streaming": {
        "youtube": "https://www.youtube.com/results?...",
        "spotify": "https://open.spotify.com/search/...",
        "deezer":  "https://www.deezer.com/search/...",
        "apple":   "https://music.apple.com/search?..."
      }
    }
  ],
  "confident": true,
  "recommendations": [
    { "track_id": "...", "title": "Save Your Tears", "artist": "The Weeknd" }
  ]
}
```

## Fonctionnalités

- **Bouton micro** — enregistrement de `UI_LISTEN_DURATION` secondes (défaut 15 s) avec décompte animé et anneau de progression
- **Drag & drop** — déposez un fichier WAV / MP3 / OGG / WebM
- **Confiance** — badge vert (certain) ou orange (top 3 affiché) selon `score[0]/score[1] ≥ UI_CONFIDENCE_RATIO`
- **Liens streaming** — YouTube, Spotify, Deezer, Apple Music (recherche, sans clé API)
- **Recommandations** — autres morceaux du même genre dans la bibliothèque
- **FR / EN** — bascule de langue dans l'entête
- **Thème sombre** — dark minimal, responsive
