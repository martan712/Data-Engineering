PYTHON ?= .venv/bin/python
BUILD ?= portable
DATASET ?=

.PHONY: setup-paper data-small verify reproduce-core paper artifact-smoke \
	data-full reproduce-full audit-results audit-evidence audit-paper \
	test-unit test-native test-integration test-artifact test-reproduction \
	test-sanitize test-sanitizer test-full native-portable clean-clone-smoke \
	research-audit

setup-paper:
	./setup.sh --full --frozen

data-small:
	$(PYTHON) -m bondmaxsim.data.fixture --output data/fixture

verify:
	$(PYTHON) -m pytest -q

test-unit:
	$(PYTHON) -m pytest -q -m unit

test-native:
	$(PYTHON) -m pytest -q -m native --native-required

test-integration:
	$(PYTHON) -m pytest -q -m integration

test-artifact:
	$(PYTHON) -m pytest -q -m artifact
	$(PYTHON) -m bondmaxsim.research_audit results
	$(PYTHON) -m bondmaxsim.research_audit governance
	$(PYTHON) -m bondmaxsim.research_audit paper

test-reproduction:
	$(PYTHON) -m pytest -q -m reproduction

test-full:
	$(PYTHON) -m pytest -q

test-sanitize:
	$(MAKE) -C cpp/per_document_oracle sanitize-check
	$(MAKE) -C cpp/wide_block_maxsim_bond sanitize-check
	$(MAKE) -C cpp/fused_panel_maxsim sanitize-check

test-sanitizer: test-sanitize

native-portable:
	$(MAKE) -C cpp/per_document_oracle BUILD=portable
	$(MAKE) -C cpp/wide_block_maxsim_bond BUILD=portable
	$(MAKE) -C cpp/fused_panel_maxsim BUILD=portable

# Deterministic data-fixture foundation; the clean-clone target below adds
# representative Stage 3--5 experiment drivers.
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
	$(PYTHON) -m bondmaxsim.research_audit results

audit-evidence:
	$(PYTHON) -m bondmaxsim.research_audit governance

audit-paper:
	$(PYTHON) -m bondmaxsim.research_audit paper

clean-clone-smoke:
	$(PYTHON) -m bondmaxsim.research_audit clean-clone

research-audit: test-full test-artifact test-native test-sanitizer clean-clone-smoke
