# Contributing

Bug reports and ideas are welcome: use the issue forms. For code changes please open an issue first so we can agree on the direction.

## Ground rules of the project
- **The output of the algorithms is deterministic** (fixed seed): the same frame always gives the same result, bit for bit. A change that alters
  the output of levels 1-5 or Symbols needs a very good reason.
- No GPU settings in the interface (the app decides by itself), no "style" knobs: the five levels are a balance of speed and quality.
- The interface is drawn pixel by pixel in the project's own style (`benvid/qtui`); no native Qt widgets.

## Running the tests
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt pytest
set QT_QPA_PLATFORM=offscreen
python -m pytest -m "not slow" -q
```
`scripts\tests.bat` offers the same as a menu. The full run (including video and process-pool tests) takes several minutes.

## Style
Commit messages in English, short and about what changed. Keep notes, scratch files and local tooling out of commits
(`tests/test_repo_clean.py` checks the obvious cases).

By contributing you agree that your contribution is licensed under the Apache License 2.0, like the rest of the project.
