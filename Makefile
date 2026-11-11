export PYTHONPATH := src

.PHONY: install train explain drift test serve docker all
install:  ; pip install -r requirements.txt
train:    ; python -m fraudshield.train
explain:  ; python -m fraudshield.explain
drift:    ; python -m fraudshield.drift
test:     ; pytest -q
serve:    ; uvicorn fraudshield.api:app --reload
docker:   ; docker build -t fraudshield . && docker run -p 8000:8000 fraudshield
all: train explain drift test
