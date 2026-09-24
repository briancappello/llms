REPO := $(shell pwd)
BIN := $(HOME)/.local/bin
UNAME_S := $(shell uname -s)

# The vLLM build and its servers are ROCm, so Linux-only.
define linux_only
	@test "$(UNAME_S)" = Linux || { echo "error: '$@' is Linux-only (vLLM ROCm); this host is $(UNAME_S)" >&2; exit 1; }
endef

.PHONY: help install uninstall test build proxy check corpus swebench-venv engine vllm vllm-reranker vllm-embeddings

help:
	@printf '%s\n' \
	  'install        install the Python CLI with the GGUF extra' \
	  'uninstall      remove the Python CLI tool' \
	  'test           run Python and Go tests' \
	  'build          build the Python wheel and source archive' \
	  'engine         build an inference engine, e.g.' \
	  '               make engine NAME=bonsai BACKEND=hip SRC=~/dev/bonsai-llama.cpp' \
	  '               make engine NAME=vulkan BACKEND=vulkan' \
	  'vllm           build pinned vLLM source for the configured ROCm GPU' \
	  'vllm-reranker  serve BAAI/bge-reranker-v2-m3 on port 8010' \
	  'vllm-embeddings serve voyageai/voyage-4-nano on port 8011' \
	  'proxy          build the optional authentication proxy' \
	  'check          run package and configuration checks' \
	  'corpus         build the benchmark code corpus' \
	  'swebench-venv  create the optional SWE-bench environment'

# Thin wrapper; bin/build-engine takes the full option set.
engine:
	@test -n "$(NAME)" -a -n "$(BACKEND)" || { echo "usage: make engine NAME=<n> BACKEND=<b> [SRC=<dir>] [REF=<ref>] [ARCH=<gfx|cc>]"; exit 1; }
	$(REPO)/bin/build-engine $(NAME) $(BACKEND) \
	  $(if $(SRC),--src $(SRC)) $(if $(REF),--ref $(REF)) $(if $(ARCH),--arch $(ARCH))

vllm:
	$(linux_only)
	$(REPO)/bin/build-vllm $(if $(REF),--ref $(REF)) $(if $(ARCH),--arch $(ARCH)) $(if $(JOBS),--jobs $(JOBS))

vllm-reranker:
	$(linux_only)
	$(REPO)/bin/run-vllm-reranker

vllm-embeddings:
	$(linux_only)
	$(REPO)/bin/run-vllm-embeddings

install:
	uv tool install --force --with gguf $(REPO)

uninstall:
	uv tool uninstall llms

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
	uv run --extra gguf llms --help >/dev/null
	uv run --extra gguf llms doctor

swebench-venv:
	cd bench/swebench && uv venv .venv --python 3.12
	uv pip install --python bench/swebench/.venv/bin/python mini-swe-agent swebench

corpus:
	python3 bench/lib/build_code_corpus.py
