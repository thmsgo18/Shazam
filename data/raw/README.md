# Local Test Audio Dataset

This directory contains the audio queries used for demos, manual checks, and report-oriented evaluation.

The files in `data/raw/` are intentionally kept out of GitHub. We only version:

- this `README.md`;
- `manifest.example.json`;
- `.gitkeep` placeholders when needed.

The actual audio files stay on each teammate's machine or in a private shared storage space.

## Why the audio is not versioned

Keeping the raw test audio outside Git is the cleanest and most professional setup for this project:

- reference excerpts may be derived from copyrighted songs;
- microphone recordings may contain personal voices, room noise, or other private context;
- audio binaries make the repository heavier for no real benefit;
- reproducibility is better achieved through a documented protocol and a manifest than through large binary blobs.

In other words: we version the dataset specification, not the media files themselves.

## Expected layout

Recommended local structure:

```text
data/raw/
├── README.md
├── manifest.example.json
├── manifest.json
├── reference_clips/
│   ├── Artist-Title-middle_15s.mp3
│   └── ...
└── mic_recordings/
    ├── Artist-Title-mic-normal-clean.mp3
    └── ...
```

Recommended roles:

- `reference_clips/`: short excerpts cut from the original track, used as clean studio-like queries.
- `mic_recordings/`: real microphone captures recorded by the team, used to test robustness in realistic conditions.
- `manifest.json`: local inventory of the files that are actually present on the machine.

## What the manifest does

`manifest.json` is the ground-truth file used by the evaluation pipeline. Each entry links one local audio file to the expected `track_id`.

Important conventions:

- `filename` must be a path relative to `data/raw/`.
- `track_id` should match the ID stored in `data/processed/metadata.parquet`.
- `artist` and `title` keep the human-readable metadata.
- `position` describes the query type and is the main structured label used by analyses.
- `duration_s` stores the clip duration in seconds.

Example:

```json
[
  {
    "filename": "reference_clips/David_Kushner-Daylight-middle_15s.mp3",
    "track_id": "18b0a73bfd72fa7c03cefd8f2d2619df",
    "artist": "David Kushner",
    "title": "Daylight",
    "position": "middle",
    "duration_s": 15.0
  },
  {
    "filename": "mic_recordings/David_Kushner-Daylight-mic-normal-clean.mp3",
    "track_id": "18b0a73bfd72fa7c03cefd8f2d2619df",
    "artist": "David Kushner",
    "title": "Daylight",
    "position": "mic_normal_clean",
    "duration_s": 20.6
  }
]
```

## Naming convention

The project does not require a single exact filename format at runtime, but using a strict convention makes the dataset understandable, keeps the manifest readable, and avoids duplicate or ambiguous names.

Use the following patterns:

- Reference clip: `ArtistSlug-TitleSlug-Position_Duration.mp3`
- Microphone recording: `ArtistSlug-TitleSlug-mic-Distance-Speech.mp3`

Examples:

- `David_Kushner-Daylight-middle_15s.mp3`
- `Loreen-Tattoo-start_5s.mp3`
- `Tyler_The_Creator-See_You_Again-middle_30s.mp3`
- `David_Kushner-Daylight-mic-close-clean.mp3`
- `David_Kushner-Daylight-mic-normal-speech.mp3`

## Filename normalization rules

### 1. Keep filenames ASCII-safe

Use only:

- letters `A-Z` and `a-z`;
- digits `0-9`;
- underscore `_`;
- hyphen `-`.

Avoid spaces, quotes, accents, slashes, commas, ampersands, parentheses, and other punctuation in filenames.

Good:

- `Tyler_The_Creator-See_You_Again-middle_15s.mp3`

Bad:

- `Tyler, The Creator - See You Again (feat. Kali Uchis) - middle 15s.mp3`

### 2. Separate semantic blocks with hyphens

Use `-` to separate the main blocks of meaning:

- artist slug;
- title slug;
- query descriptor.

Examples:

- `ArtistSlug-TitleSlug-middle_15s.mp3`
- `ArtistSlug-TitleSlug-mic-normal-clean.mp3`

### 3. Use underscores inside artist and title slugs

Inside the artist and title blocks, replace spaces with `_`.

Examples:

- `David Kushner` -> `David_Kushner`
- `Harry Styles` -> `Harry_Styles`
- `See You Again` -> `See_You_Again`

This keeps filenames readable while still being shell-safe.

Recommended case policy:

- keep a readable case style based on the source metadata;
- in practice, capitalized words such as `David_Kushner` or `See_You_Again` are preferred in this project;
- do not mix several styles for the same dataset, for example `see_you_again` in one file and `See_You_Again` in another.

### 4. Remove accents and most punctuation

Normalize names to a plain ASCII slug:

- transliterate non-ASCII characters to ASCII when possible;
- remove commas, apostrophes, quotes, and periods;
- replace `/` and `&` with a separator only if needed for readability;
- collapse repeated separators;
- trim separators at the beginning or the end of a slug.

Examples:

- `P!nk` -> `Pink`
- `AC/DC` -> `AC_DC`
- `Tyler, The Creator` -> `Tyler_The_Creator`
- `Don't Start Now` -> `Dont_Start_Now`

If a title contains decorative punctuation, simplify it instead of preserving the original typography.

Examples:

- `Starboy - Live` -> `Starboy_Live`
- `Song Title!!!` -> `Song_Title`

### 5. Deterministic normalization recipe

For each artist or title block, apply the same sequence every time:

1. Start from the human-readable metadata.
2. Remove optional wording that should not appear in the filename slug.
3. Replace accented or special characters with their ASCII equivalent when possible.
4. Replace spaces with `_`.
5. Remove punctuation that does not help identify the track.
6. Replace remaining unsafe separators such as `/` with `_`.
7. Collapse repeated `_` or `-`.
8. Trim leading and trailing separators.

Example:

- Raw title: `See You Again (feat. Kali Uchis)`
- Simplified title block: `See You Again`
- Normalized title slug: `See_You_Again`

### 6. Keep the filename short and stable

The manifest already stores the full human-readable metadata. The filename should be a stable identifier, not a full sentence.

Recommended simplifications:

- remove feature mentions such as `(feat. ...)` or `(with ...)`;
- remove version notes such as `(Remastered 2011)` when they are not needed to disambiguate the track;
- keep only the canonical track title in the slug.

Examples:

- `Calm Down (with Selena Gomez)` -> `Calm_Down`
- `See You Again (feat. Kali Uchis)` -> `See_You_Again`

Keep the full original title in the manifest `title` field.

### 7. Handle collisions explicitly

If two different files would produce the same normalized name, add a short disambiguation token at the end of the title block or query descriptor.

Examples:

- `Artist-Title-radio_edit-middle_15s.mp3`
- `Artist-Title-live-middle_15s.mp3`
- `Artist-Title-take2-mic-normal-clean.mp3`

Do not solve collisions by inventing inconsistent naming schemes. Add one short, stable token and keep the rest of the pattern unchanged.

### 8. Do not encode too much metadata in the filename

The filename should identify the file, not replace the manifest.

Use the filename to encode only:

- which track it belongs to;
- whether it is a studio excerpt or a microphone recording;
- the main capture condition.

Store the rest in `manifest.json`.

## Position and condition labels

The evaluation code primarily reads the `position` field from the manifest. For microphone files, it can also infer some information from the filename as a fallback. Because of that, it is best to keep both aligned.

Recommended values for studio reference clips:

- `start`
- `first-quarter`
- `middle`
- `third-quarter`
- `end`

Recommended values for microphone recordings:

- `mic_close_clean`
- `mic_close_speech`
- `mic_normal_clean`
- `mic_normal_speech`
- `mic_far_clean`

Recommended filename forms for the same microphone conditions:

- `...-mic-close-clean.mp3`
- `...-mic-normal-speech.mp3`
- `...-mic-far-clean.mp3`

Rule of thumb:

- `position` is the source of truth for structured evaluation labels;
- the filename should mirror that structure for readability and manual inspection.

## Practical normalization workflow

When you add a new local test file, use this process:

1. Start from the real metadata: artist, title, expected `track_id`.
2. Build an ASCII-safe artist slug.
3. Build an ASCII-safe title slug.
4. Add the query descriptor. Use `middle_15s`, `start_5s`, and similar labels for reference clips. Use `mic-close-clean`, `mic-normal-speech`, and similar labels for microphone recordings.
5. Save the file in the appropriate subdirectory.
6. Add the corresponding entry to `manifest.json`.
7. Check that the manifest path exactly matches the relative file path under `data/raw/`.

## Examples of normalized names

Raw metadata:

- Artist: `Tyler, The Creator`
- Title: `See You Again (feat. Kali Uchis)`
- Query type: reference clip
- Position: `middle`
- Duration: `15s`

Normalized filename:

- `Tyler_The_Creator-See_You_Again-middle_15s.mp3`

Raw metadata:

- Artist: `Rema`
- Title: `Calm Down (with Selena Gomez)`
- Query type: microphone recording
- Distance: `normal`
- Condition: `speech`

Normalized filename:

- `Rema-Calm_Down-mic-normal-speech.mp3`

## Recommended team workflow

1. Copy `manifest.example.json` to `manifest.json` on each machine.
2. Store the real audio locally in `reference_clips/` and `mic_recordings/`.
3. Keep the same normalized filenames across the whole team.
4. Keep the same `track_id` values across all manifests.
5. Share the real audio only through private storage.
6. Commit only documentation and manifest examples to Git.

## Common mistakes to avoid

- Do not commit `manifest.json` if it lists private local files.
- Do not use absolute paths in the manifest.
- Do not leave spaces or punctuation-heavy titles in filenames.
- Do not make the filename and `position` disagree.
- Do not use different naming styles for the same track across teammates.

## Minimal rule set

If you want one short policy to follow, use this:

- keep audio files out of Git;
- store paths relative to `data/raw/` in `manifest.json`;
- use normalized ASCII filenames;
- use `ArtistSlug-TitleSlug-middle_15s.mp3` for reference clips;
- use `ArtistSlug-TitleSlug-mic-normal-clean.mp3` for microphone recordings;
- keep `position` and filename consistent.
