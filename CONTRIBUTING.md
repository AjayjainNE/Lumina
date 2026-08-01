## Contributing to LUMINA

Thank you for your interest in contributing!

### Setup

```bash
git clone https://github.com/yourusername/lumina-doc-intelligence.git
cd lumina-doc-intelligence
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Branch Strategy

```
main        ← stable, tagged releases
develop     ← integration branch
feature/*   ← new features
fix/*       ← bug fixes
nb/*        ← notebook additions
```

### Running Tests

```bash
pytest tests/ -v --tb=short       # all tests
pytest tests/ -k TestCGRRouter    # specific class
pytest tests/ -k TestLLMJudge     # judge tests only
```

### Adding a New Agent

1. Create `src/agents/your_agent.py` extending `BaseAgent`
2. Implement `process(chunk, document_id) -> AgentOutput`
3. Register in `api/app.py` under `agents = {...}`
4. Add entry to `config/model_config.yaml`
5. Write tests in `tests/test_lumina.py`

### Improving CGR Routing

The core algorithm is in `src/routing/cgr_algorithm.py`.
Key areas for improvement:
- `ENSEMBLE_THRESHOLD` — tune the confidence cutoff
- `ENTROPY_COEFF` — exploration bonus coefficient
- `ConfidenceEstimator` — try deeper / attention-based architectures
- `_compute_gae` — experiment with different λ values

### Code Style

- `black` for formatting, `ruff` for linting
- Type hints required for all public functions
- Docstrings in Google format
- All new modules must have an `if __name__ == "__main__":` smoke test

### Pull Request Checklist

- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] `SKILL.md` / docs updated if API changes
- [ ] Notebook added/updated if new end-to-end capability
- [ ] `config/model_config.yaml` updated if new model added
