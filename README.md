# MMAT_signal_project

## 🛠️ Environment Setup

To run this project, choose **one** of the following setup methods:

---

### Option 1: Conda Environment (Recommended for Windows + TA-Lib)

This method is ideal for full compatibility with **TA-Lib** and better dependency management.

```
conda env create -f environment.yml
conda activate mmat-env
```

**Note:** On Windows, `ta-lib` may not install automatically via `pip`.  
To manually install it:

1. Download the appropriate `.whl` file from:  
   👉 https://github.com/mrjbq7/ta-lib/releases

   *(Example for Python 3.10: `ta_lib-0.6.4-cp310-cp310-win_amd64.whl`)*

2. Install it after activating the environment:

```
pip install wheels/ta_lib-*.whl
```

3. Verify it worked:

```
python -c "import talib; print(talib.__version__)"
```

---

### 💡 Option 2: Virtualenv + pip (Linux/macOS or minimal setups)

If you're not using Conda, create and activate a virtual environment:

```bash
python -m venv venv

# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Then install core dependencies:
pip install -r requirements.txt
```

**Important:**  
TA-Lib must still be installed manually via `.whl` (Windows) or system libraries (Linux/macOS).  
See `requirements.txt` for installation tips.

---
