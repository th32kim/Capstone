.PHONY: setup lint test detect gate validate process ingest eval figures clean

setup:
	python -m pip install -U pip
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@command -v tesseract >/dev/null || echo "WARNING: tesseract not found (FS8/OCR will be disabled)"
	@echo "setup ok"

lint:
	ruff check hindsight tests
	black --check hindsight tests
	mypy --strict hindsight/contracts.py

test:
	pytest -q

# the whole funnel, one clip
pipeline: 
	hindsight detect   $(CLIP)
	hindsight gate     $(CLIP)
	hindsight validate $(CLIP)
	hindsight process  $(CLIP)
	hindsight ingest   $(CLIP)

eval:
	hindsight eval gating
	hindsight eval cascade
	hindsight eval retrieval
	hindsight eval latency
	hindsight eval index

# regenerate every report figure FROM MEASUREMENTS. zero hand-entered numbers.
figures: eval
	hindsight eval figures
	@echo "-> out/figures/  and  out/reports/measurements.md"

clean:
	rm -rf out/
