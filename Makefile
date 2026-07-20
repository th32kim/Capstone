.PHONY: setup prep-720p lint test detect gate validate process ingest eval figures clean

setup:
	python -m pip install -U pip
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@mkdir -p models/face
	@test -f models/face/face_detection_yunet_2023mar.onnx || \
	  curl -fsSL -o models/face/face_detection_yunet_2023mar.onnx \
	  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
	@command -v tesseract >/dev/null || echo "WARNING: tesseract not found (FS8/OCR will be disabled)"
	@echo "setup ok"

prep-720p:      ## downscale a 4K clip to the 720p capture target: make prep-720p SRC=in.mp4 DST=data/corpus/x.mp4
	python scripts/to720p.py $(SRC) $(DST)

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
