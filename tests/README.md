Test suite for the project.

The suite is based on `unittest` and is meant to stay fast, local, and reproducible. It focuses on the Python/backend codebase: pipeline logic, CLI, API, ingestion, maintenance, evaluation, and persistence helpers.

Layout:
- `tests/helpers/`: shared factories for synthetic audio, metadata, and SQLite fixtures.
- `tests/fixtures/`: small versioned fixtures used by tests.
- `tests/unit/`: isolated tests for modules and helper functions.
- `tests/integration/`: lightweight end-to-end checks across module boundaries, CLI routing, and API routes.

Recommended commands:

```bash
python manage.py test
python manage.py test --buffer
python manage.py test --unit
python manage.py test --integration --failfast
```

Direct `unittest` discovery also works:

```bash
python -m unittest discover -s tests -v
```

Useful options:
- `--buffer`: hide prints/logs for passing tests and keep failures readable.
- `--failfast`: stop on the first failure.
- `--quiet`: reduce verbosity.
- `--path PATH`: run one specific test folder.
- `--pattern GLOB`: restrict discovery to a filename pattern.

Notes:
- The suite heavily covers the Python/backend project.
- The React frontend under `webapp/frontend/src/` is not covered by this test suite yet.
- Integration tests rely on synthetic fixtures and lightweight temporary data rather than a large music dataset.
