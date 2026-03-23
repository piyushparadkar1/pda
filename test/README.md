# Test directory compatibility note

The canonical automated test suite for this repo lives in `tests/`.

This `test/` folder exists as a compatibility/visibility alias for workflows or reviewers
expecting a singular `test` directory in the branch layout.

Use the canonical suite with:

```bash
python -m unittest discover -s tests -v
```
