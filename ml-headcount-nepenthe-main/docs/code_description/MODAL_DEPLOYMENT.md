# Modal Integration Guide

This document explains how to use the `ml_headcount` package with Modal Labs for cloud-based ML processing using ephemeral apps.

## Overview

The Modal integration allows us to run computationally expensive operations like KeyBERT keyword extraction and clustering on Modal's GPU-enabled cloud infrastructure using ephemeral apps. The setup integrates with Hamilton's data loading and execution framework, providing a seamless way to run remote processing tasks without explicit deployment.

## Project Structure

```
ml-headcount/
├── src/
│   └── ml_headcount/
│       ├── __init__.py
│       ├── modal_functions.py     # Modal app and function implementations
│       └── ... (other modules)
├── pyproject.toml                 # Package configuration
└── MODAL_DEPLOYMENT.md           # This file
```

## Key Components

### 1. Package Configuration (`pyproject.toml`)

The package is properly configured as a Python package with:
- `packages = [{include = "ml_headcount", from = "src"}]` - Tells Python where to find the package
- All dependencies listed in `dependencies` array
- `uv_build` as the build backend for compatibility with Modal's `uv_sync()`

### 2. Modal App Definition (`src/ml_headcount/modal_functions.py`)

This file contains both the Modal app definition and function implementations:

```python
import modal

# Create the main app
app = modal.App("ml-headcount")

# Function to download models (no GPU needed)
def download_models():
    """Download and cache models during image build"""
    from sentence_transformers import SentenceTransformer
    
    # Download both models
    SentenceTransformer("all-MiniLM-L6-v2")
    SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    print("✅ Models downloaded and cached successfully")

# Define the image with pre-downloaded models
image = (
    modal.Image
        .debian_slim(python_version="3.11")
        .uv_sync()
        .run_function(download_models)  # No GPU needed for download
        .add_local_python_source("ml_headcount")
)

# Decorate and define functions
@app.function(image=image, gpu="A10G", memory=16384, cpu=4, timeout=3600)
def run_keybert_extraction(cv_data, model_name="all-MiniLM-L6-v2", ...):
    """KeyBERT implementation with GPU acceleration"""
    # Function logic here
    pass

@app.function(image=image, gpu="A10G", memory=8192, cpu=2, timeout=1800)
def run_keyword_clustering(discriminative_keywords, sentence_model, ...):
    """Keyword clustering with GPU acceleration"""
    # Function logic here
    pass

# Local functions to call Modal functions
def call_modal_keybert_extraction(cv_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Call Modal KeyBERT extraction from local code."""
    function = modal.Function.from_name("ml-headcount", "run_keybert_extraction")
    return function.remote(cv_data=cv_data, **kwargs)
```

## Hamilton Integration

### 3. Hamilton Pipeline Integration (`test_modal_integration.py`)

The current approach integrates Modal functions with Hamilton's data loading and execution framework using ephemeral apps:

```python
from ml_headcount.hamilton_pipeline import HamiltonMLHeadcountPipeline
from ml_headcount.modal_functions import app as modal_app

# Initialize pipeline with remote processing enabled
pipeline = HamiltonMLHeadcountPipeline(
    data_dir="data",
    output_dir="test_outputs",
    use_remote=True,  # Enable Modal
    disable_cache=True  # Disable cache to avoid issues
)

# Execute with ephemeral app
with modal_app.run():
    results = pipeline.dr.execute(["test_keybert_modal"])
```

This approach provides:
- **Seamless Integration**: Modal functions are called through Hamilton's execution framework
- **Data Management**: Hamilton handles data loading, caching, and serialization
- **Subgraph Execution**: Run specific parts of the pipeline on Modal
- **Error Handling**: Hamilton provides robust error handling and logging
- **Ephemeral Apps**: No explicit deployment required - functions run on-demand

## Usage Methods

### Method 1: Ephemeral Apps (Recommended)

Use ephemeral apps with `with app.run()` - no deployment required:

```python
from ml_headcount.modal_functions import app as modal_app

# Run functions with ephemeral app
with modal_app.run():
    result = run_keybert_extraction.remote(cv_data=your_data)
```

### Method 2: Hamilton Integration Testing

Test the complete integration using the Hamilton pipeline:

```bash
# Run the integration test
uv run test_modal_integration.py
```

This will:
1. Initialize the Hamilton pipeline with `use_remote=True`
2. Execute the `test_keybert_modal` subgraph using ephemeral apps
3. Display comprehensive results and statistics

## Key Design Principles

### 1. Package Inclusion

Modal includes the `ml_headcount` package using:
- `uv_sync()` reads the `pyproject.toml` and includes all dependencies
- `add_local_python_source("ml_headcount")` explicitly includes the local package
- The package structure follows Python standards

### 2. Unified Function Definition

Functions are defined in a single layer:
- **Implementation layer** (`modal_functions.py`): Modal-decorated functions with business logic

This unified approach allows:
- Easy testing of business logic locally
- Clean Modal integration without separate deployment files
- Reusable function implementations
- Ephemeral app usage with `with app.run()`

### 3. Image Management

All functions share the same image definition:
```python
image = (
    modal.Image
        .debian_slim(python_version="3.11")
        .uv_sync()
        .add_local_python_source("ml_headcount")
)
```

This ensures:
- Consistent environment across all functions
- Automatic dependency resolution from `pyproject.toml`
- Explicit inclusion of the local package
- Efficient image caching

## Available Functions

### 1. `test_wrapper()`
- **Purpose**: Verify Modal deployment and package imports
- **Resources**: CPU only, 60s timeout
- **Usage**: `uv run modal run -m ml_headcount.deploy::test_wrapper`

### 2. `run_keybert_extraction()`
- **Purpose**: Extract keywords using KeyBERT with GPU acceleration
- **Resources**: A10G GPU, 16GB RAM, 4 CPU cores, 1 hour timeout
- **Input**: DataFrame with 'category' and 'processed_text' columns
- **Output**: Dictionary with extracted keywords and scores

### 3. `run_keyword_clustering()`
- **Purpose**: Cluster extracted keywords using HDBSCAN
- **Resources**: A10G GPU, 8GB RAM, 2 CPU cores, 30 min timeout
- **Input**: Discriminative keywords dictionary and sentence model
- **Output**: Dictionary with clustering results

## Testing the Integration

### 1. Basic Test
```python
from ml_headcount.modal_functions import app as modal_app, test_wrapper

# Run with ephemeral app
with modal_app.run():
    result = test_wrapper.remote()
```

Expected output:
```
🚀 Starting test_wrapper function on Modal...
✅ Successfully imported ml_headcount modules!
🎉 Test completed successfully!
```

### 2. Verify Package Inclusion
The logs should show:
- `🔨 Created mount PythonPackage:ml_headcount` - Package is included
- `✅ Successfully imported ml_headcount modules!` - Imports work
- Package path: `/root/ml_headcount` - Correct installation location

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure `pyproject.toml` has correct package configuration
   - Verify `packages = [{include = "ml_headcount", from = "src"}]`

2. **Import errors in Modal functions**
   - Use relative imports within the package: `from .module import function`
   - Ensure all dependencies are in `pyproject.toml`

3. **Function not accessible with ephemeral apps**
   - Use `with app.run():` context manager
   - Ensure functions are properly decorated with `@app.function`

### Debugging

1. **Check package inclusion**:
   ```python
   from ml_headcount.modal_functions import app as modal_app, test_wrapper
   with modal_app.run():
       result = test_wrapper.remote()
   ```

2. **View function logs**:
   - Logs appear in real-time during execution
   - Use `print()` statements for debugging

3. **Verify ephemeral app**:
   - Functions run on-demand with `with app.run()`
   - No explicit deployment required

## Best Practices

1. **Use ephemeral apps** with `with app.run():` - no deployment required
2. **Keep functions unified** - combine Modal decorators with business logic
3. **Test locally first** - ensure functions work without Modal
4. **Use appropriate resources** - match GPU/memory to workload
5. **Handle timeouts** - set reasonable timeouts for long-running functions
6. **Log everything** - use print statements for debugging and monitoring

## Integration with Local Code

To call Modal functions from local code using ephemeral apps:

```python
from ml_headcount.modal_functions import app as modal_app, run_keybert_extraction

# Use ephemeral app
with modal_app.run():
    result = run_keybert_extraction.remote(cv_data=your_data)
```

Or use the helper functions in `modal_functions.py`:

```python
from ml_headcount.modal_functions import call_modal_keybert_extraction

# Call with automatic serialization
result = call_modal_keybert_extraction(cv_data=your_data)
```

## References

- [Modal Hello World Example](https://modal.com/docs/examples/hello_world)
- [Modal Project Structure Guide](https://modal.com/docs/guide/project-structure)
- [Modal Image Reference](https://modal.com/docs/reference/modal.Image#uv_sync)
