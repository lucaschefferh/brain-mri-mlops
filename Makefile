.PHONY: setup lock lint test baseline

setup:  ## cria o venv e instala as dependências travadas
	uv venv --python 3.11 .venv
	.venv/bin/python -m ensurepip --upgrade || true
	uv pip install --python .venv/bin/python -r requirements.txt

lock:   ## regenera requirements.txt a partir de requirements.in
	uv pip compile requirements.in -o requirements.txt --python-version 3.11

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q -m "not slow"

baseline:  ## reproduz a inferência do checkpoint B2 e compara com o baseline gravado
	.venv/bin/python src/inference.py --checkpoint checkpoints/unetpp_efficientnet-b2_v1_best.pth
