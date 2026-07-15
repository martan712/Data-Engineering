PYTHON ?= .venv/bin/python
BUILD ?= portable
DATASET ?=

.PHONY: setup-paper data-small verify reproduce-core paper artifact-smoke \
	data-full reproduce-full audit-results audit-evidence

setup-paper:
	./setup.sh --full --frozen

data-small:
	$(PYTHON) -m bondmaxsim.data.fixture --output data/fixture

verify:
	$(PYTHON) -m pytest -q

# Stage 2 scaffold: validates the deterministic fixture. Stage 4 extends this
# target with representative mechanism and system experiment drivers.
reproduce-core: data-small
	$(PYTHON) -m bondmaxsim.data.fixture --verify data/fixture

# Publication generation remains separate from expensive experiment execution.
paper:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

artifact-smoke: data-small verify reproduce-core paper

data-full:
	@echo "data-full is blocked until immutable dataset/model revisions are resolved in Stage 6." >&2
	@echo "Expected scope: four BEIR corpora, multi-GB embeddings/indexes, and resumable dataset selection." >&2
	@exit 2

reproduce-full:
	@echo "reproduce-full is blocked until the Stage 7 code/data freeze and final-run manifest." >&2
	@exit 2

audit-results:
	@echo "audit-results becomes available with the Stage 3 schema registry." >&2
	@exit 2

audit-evidence:
	@echo "audit-evidence becomes available with the Stage 5 evidence manifest." >&2
	@exit 2
