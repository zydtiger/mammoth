# Mammoth

Mammoth is project-independent infrastructure for running, training, logging,
and monitoring AI workloads. It provides operational mechanics without owning
model architecture, dataset, loss, or domain-metric semantics.

## Development

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python -c "import mammoth; print(mammoth.__version__)"
```

Repository documentation begins with [AGENTS.md](AGENTS.md).
