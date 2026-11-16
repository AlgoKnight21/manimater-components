# Community components

One file per component. Drop `your_component.py` in this folder and `load_all()` registers it: nothing else to wire up.

1. Copy [`_template.py`](_template.py) to `your_component.py` and edit it.
2. Add tests for your pure-Python logic in `tests/community/test_your_component.py`. The generic checks (the example renders, the rules below) already run on every file in this folder.
3. Run `python -m pytest` and `python -m manimate.preview your_component` to see it.
4. Open a pull request. CI regenerates `catalog.json` and checks it.

Not a programmer? [Generate it with an LLM](../../../docs/GENERATE_WITH_AN_LLM.md) and open an issue instead.
