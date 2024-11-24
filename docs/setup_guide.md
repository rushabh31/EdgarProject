# Setup Guide for Using the Makefile

This guide provides a detailed overview of how to set up your environment, modify necessary files, and execute Python tasks using the provided Makefile. **For the fastest and easiest setup, use the `make all` command as described below.**

---

## Prerequisites
Before starting, ensure the following are installed on your system:
- **Python 3.10 or above**
- **make utility**
- **pip** (Python package installer)
- **pre-commit hooks** (`pip install pre-commit` if not already installed)

---

## Instructions

### 1. Clone the Repository
Start by cloning the repository and navigating into the project directory:
```bash
git clone[ <repository_url>](https://github.com/rushabh31/EdgarProject/tree/main)
cd EdgarProject
```

### 2. Verify Python Installation
Ensure Python 3.10 or above is installed. The Makefile will check for Python 3.10.13 at `/opt/python/3.10.13` by default. If not found, it will use your system's `python3`.

---

### 3. Recommended: Use `make all`

To set up everything with a single command, run:
```bash
make all
```

The `make all` command will:
1. Create a virtual environment named `edgar-venv`.
2. Install all dependencies from `requirements.txt`.
3. Set up pre-commit hooks.
4. Register the virtual environment as a Jupyter kernel.
5. Modify the `outlines.py` file to update `device='auto'`.

**This is the most efficient way to get started and ensures all steps are executed properly.**

---

### 4. Alternative Commands

If you prefer a step-by-step approach, use the individual commands described below:

#### **Show Help**
To list available Makefile commands:
```bash
make help
```

#### **Set Up the Virtual Environment**
To create the virtual environment and install dependencies, run:
```bash
make venv
```

#### **Modify `outlines.py`**
To update the `outlines.py` file and set `device='auto'`, run:
```bash
make modify_outlines
```

---

### 5. Activate the Virtual Environment
After setup, activate the virtual environment:
```bash
source edgar-venv/bin/activate
```

---

### 6. Run Python Scripts
Once the virtual environment is activated, you can execute the provided scripts:

#### **Run Task 1**
```bash
python3 task_1_engineering.py
```

#### **Run Task 2**
```bash
python3 task_2_genai.py
```

---

## Troubleshooting
- **Setup Errors:** If `make all` fails, check the specific error message and verify the prerequisites (e.g., Python version, dependencies in `requirements.txt`).
- **File Modification Issues:** Ensure that `outlines.py` exists in the path `edgar-venv/lib/python3.10/site-packages/langchain_community/llms/outlines.py`.
- **Virtual Environment Issues:** Ensure the virtual environment is activated correctly using `source edgar-venv/bin/activate`.

---
