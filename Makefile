.PHONY: smoke all report test lint fmt
smoke:  ; uv run python -m src.bench --datasets scifact,fiqa --arms all --policies all --queries 25 --split dev
all:    ; uv run python -m src.bench --datasets scifact,fiqa --arms all --policies all --split test
report: ; uv run python -m src.report
test:   ; uv run pytest
lint:   ; uv run ruff check . && uv run ruff format --check .
fmt:    ; uv run ruff format .
