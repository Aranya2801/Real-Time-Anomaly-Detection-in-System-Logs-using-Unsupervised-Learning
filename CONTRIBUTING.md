# Contributing to LogSentinel AI

Thank you for your interest in contributing! This document outlines standards and workflows.

## Development Setup

```bash
git clone https://github.com/your-org/LogSentinel-AI
cd LogSentinel-AI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
cp .env.example .env
```

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable production releases |
| `develop` | Integration branch |
| `feature/*` | New features |
| `fix/*` | Bug fixes |
| `research/*` | Experimental ML work |

## Pull Request Guidelines

1. Branch from `develop`, not `main`
2. Write tests for all new code (target >80% coverage)
3. Run full test suite before submitting: `pytest tests/ -v`
4. Add docstrings to all public functions and classes
5. Update `docs/` if adding features
6. Link related GitHub issue in PR description

## Code Standards

```python
# Good: Type hints, docstrings, descriptive names
async def detect_anomalies(
    features: np.ndarray,
    threshold: float = 0.7,
) -> List[AnomalyResult]:
    """
    Run ML ensemble on feature matrix.
    
    Args:
        features: [batch_size, feature_dim] normalized feature matrix
        threshold: Minimum score to classify as anomaly
        
    Returns:
        List of AnomalyResult objects with scores and explanations
    """
    ...
```

## Adding a New ML Algorithm

1. Create `ml_models/algorithms/your_algorithm.py`
2. Implement `anomaly_score(x) -> Tensor` method returning scores in [0,1]
3. Implement `calibrate(normal_data, percentile)` for threshold calibration
4. Add to `ml_models/ensemble/adaptive_ensemble.py` MODEL_NAMES
5. Register in `backend/ml_engine/model_registry.py`
6. Add weight in `backend/core/config.py`
7. Write tests in `tests/unit/test_ml/`
8. Add documentation in `docs/ml_models/`

## Research Contributions

For novel research contributions:
1. Add to `ml_models/research/` directory
2. Include paper reference in docstring
3. Provide benchmark comparison in `docs/research/benchmarks.md`
4. Add Jupyter notebook demo in `notebooks/`

## Running Tests

```bash
# All tests
pytest tests/ -v

# Only unit tests (fast)
pytest tests/unit/ -v

# ML model tests
pytest tests/unit/test_ml/ -v -m ml

# With coverage
pytest tests/ --cov=backend --cov=ml_models --cov-report=html
```

## Commit Message Format

```
type(scope): Short description (50 chars max)

Longer description if needed. Wrap at 72 characters.

Refs: #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `research`

## Security Issues

**Do NOT open public GitHub issues for security vulnerabilities.**
Email: security@logsentinel.ai (PGP key in SECURITY.md)
