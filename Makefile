.PHONY: install test run serve health docker

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

run: serve

serve:
	python3 -m armadacrew serve --host 127.0.0.1 --port 8080

health:
	python3 -m armadacrew health

docker:
	docker compose up --build
