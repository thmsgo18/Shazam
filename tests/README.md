Test suite for the project.

Layout:
- `tests/helpers/`: shared factories for audio files, metadata and SQLite fixtures.
- `tests/unit/`: fast unit tests with mocks and temporary files.
- `tests/integration/`: lightweight end-to-end checks across modules and CLI/API boundaries.
- `tests/fixtures/`: small text fixtures versioned with the repository.

Recommended command:

```bash
python3 -m unittest discover -s tests -v
```
