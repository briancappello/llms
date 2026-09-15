REPO := $(shell pwd)
BIN := $(HOME)/.local/bin

.PHONY: help install uninstall test build proxy check corpus swebench-venv

help:
	@printf '%s\n' \
	  'install        install the Python CLI with the GGUF extra' \
	  'uninstall      remove the Python CLI tool' \
	  'test           run Python and Go tests' \
	  'build          build the Python wheel and source archive' \
	  'proxy          build the optional authentication proxy' \
	  'check          run package and configuration checks' \
	  'corpus         build the benchmark code corpus' \
	  'swebench-venv  create the optional SWE-bench environment'

install:
	uv tool install --force --with gguf $(REPO)

uninstall:
	uv tool uninstall llama-swap-manager

test:
	uv run --extra gguf python -m unittest discover -s tests -v
	cd proxy && go test ./...
	cd proxy && go vet ./...

build:
	rm -rf dist
	uv build --out-dir dist

proxy:
	cd proxy && go build -o $(BIN)/llama-swap-auth .

check:
	uv run --extra gguf llm --help >/dev/null
	uv run --extra gguf llm doctor

swebench-venv:
	cd bench/swebench && uv venv .venv --python 3.12
	uv pip install --python bench/swebench/.venv/bin/python mini-swe-agent swebench

corpus:
	python3 bench/lib/build_code_corpus.py
