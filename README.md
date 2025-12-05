## Setup Instructions

To ensure a reproducible environment, please follow these steps carefully.

### 1. Prerequisite: Check Your Python Version

This project requires a specific version of Python to ensure all dependencies work correctly.

**Required Python version: `>=3.9` and `<3.13` (e.g., 3.9, 3.10, 3.11).**

Before you begin, check your default `python3` version by running:

```bash
python3 --version
```

- **If your version is compatible** (e.g., it shows `Python 3.10.9`), you can proceed to the next step.
- **If your version is incompatible** (e.g., it shows `Python 3.8.10`), you must install a compatible version. The recommended way to manage multiple Python versions is by using [pyenv](https://github.com/pyenv/pyenv).

---

### 2. (Recommended) Using `pyenv` to Set the Correct Version

If you have `pyenv` installed, you can easily install and set the correct Python version for this project.

```bash
# Install a compatible Python version (if you don't have one), this may take a minute or so.
pyenv install 3.11.5

# Set the local Python version for this project directory
# This creates a `.python-version` file in the repo
pyenv local 3.11.5
```

Now, your shell will automatically use Python 3.11.5 whenever you are in this directory. You can confirm by running `python --version`.

### 3. Install `uv` package manager

This installs the `uv` package manager which will be used to install the environment and required dependencies, ensuring code reproducibility.

```bash
pip install uv
```

### 4. Install Dependencies

This command reads the `pyproject.toml` file and installs all required libraries into a virtual environment created in `.venv`.

```bash
uv sync
```

### 5. Activate the virutal environment.

This command will activate the virtual environment ensuring that all packages installed are available at runtime.

```bash
# Activate it (on macOS/Linux)
source .venv/bin/activate

# Activate it (on Windows PowerShell)
# .\.venv\Scripts\Activate.ps1
```

Your command prompt should now show `(.venv)` at the beginning.

### 6. You're ready to go!

You can now run the scripts:

```bash
# To run a script
python scripts/my_analysis_script.py
```
