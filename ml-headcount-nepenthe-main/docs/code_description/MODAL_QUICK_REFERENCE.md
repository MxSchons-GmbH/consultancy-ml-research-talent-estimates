# Modal Quick Reference

## Essential Usage

### Ephemeral Apps (No Deployment Required)
```python
from ml_headcount.modal_functions import app as modal_app, test_wrapper, run_keybert_extraction

# Test the integration
with modal_app.run():
    result = test_wrapper.remote()

# Run KeyBERT extraction (with data)
with modal_app.run():
    result = run_keybert_extraction.remote(cv_data=your_data)
```

### Hamilton Integration
```python
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
from ml_headcount.modal_functions import app as modal_app

# Initialize pipeline with remote processing
pipeline = HamiltonMLHeadcountPipeline(
    data_dir="data",
    output_dir="outputs",
    use_remote=True  # Enable Modal
)

# Execute with ephemeral app
with modal_app.run():
    results = pipeline.dr.execute(["cv_analysis"])
```

## Key Files

- `src/ml_headcount/modal_functions.py` - Modal app and function implementations
- `pyproject.toml` - Package configuration
- `MODAL_DEPLOYMENT.md` - Detailed documentation

## Troubleshooting

- **Module not found**: Check `pyproject.toml` package configuration
- **Function not accessible**: Use `with app.run():` context manager
- **Import errors**: Use relative imports within package
- **Timeout errors**: Increase timeout in function decorator

## Quick Test

```python
from ml_headcount.modal_functions import app as modal_app, test_wrapper

# Verify everything works
with modal_app.run():
    result = test_wrapper.remote()
```

Should show:
```
✅ Successfully imported ml_headcount modules!
🎉 Test completed successfully!
```
