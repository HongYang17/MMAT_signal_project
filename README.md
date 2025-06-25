# MMAT_signal_project

## Environment Setup
To run this project, choose one of the following methods:

---
### Option 1: Conda Environment 
(Recommended for TA-Lib users)
```
conda env create -f environment.yml
conda activate mmat-env
````
This will install all Python dependencies, including native TA-Lib support via conda-forge.

---
### Option 2: Virtualenv + pip 
(if not using Conda)
```
python -m venv venv
# Activate the environment:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Then install dependencies:
pip install -r requirements.txt
```
Note: TA-Lib must be installed manually if using pip (see comments in requirements.txt).