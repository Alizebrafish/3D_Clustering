# F3DCA: 3D Trajectory Clustering & Deep Autoencoder Analysis

The source code for the primary 3D clustering workflow and the supplementary ARI/NMI external-validation analysis is available in this repository. A representative XYZ-coordinate dataset from experimental groups is provided to demonstrate software installation, required input structure, and workflow execution.

---

## Visual Workflow & GUI Overview

<p align="center">
  <img width="1167" height="739" alt="Application Overview" src="https://github.com/user-attachments/assets/69200a91-6076-45b2-8973-3bc827cc7f0b" />
</p>

<p align="center">
  <img width="2850" height="1215" alt="Workflow Overview" src="https://github.com/user-attachments/assets/9a6076bd-00c6-4e13-addf-7559e2f28e31" />
</p>

<p align="center">
  <img width="2060" height="1349" alt="Cluster Distribution" src="https://github.com/user-attachments/assets/560995de-2772-4319-b455-e5c2bce3b841" />
</p>

<p align="center">
  <img width="1164" height="726" alt="Trajectory Results" src="https://github.com/user-attachments/assets/e80c5290-082e-4929-b691-2c6274d57848" />
</p>

<p align="center">
  <img width="1448" height="1086" alt="F3DCA GUI" src="https://github.com/user-attachments/assets/f49f7df9-9132-4b89-bc06-a69b2d9678d1" />
</p>

Additional documentation on generated figures can be found in [README_figures.md](https://github.com/user-attachments/files/27585786/README_figures.md).

---

## System Requirements

- **Operating System:** Windows 10/11, macOS (Intel/Apple Silicon), or Linux (Ubuntu 20.04+)
- **Python Version:** 3.10 to 3.12 (Python 3.10 recommended)
- **Disk Space:** ~2 GB for virtual environment dependencies

---

## Installation & Launch Methods

Choose **one** of the three methods below that fits your platform:

### Method 1: One-Click Automatic Setup (Windows) — *Easiest*

No command-line setup is required. The provided batch script handles virtual environment creation, pip upgrades, dependency installation, and application launching automatically.

1. Download and extract the repository ZIP file (or clone the repository).
2. Ensure you have standard [Python](https://www.python.org/downloads/) installed and checked **"Add python.exe to PATH"** during installation.
3. Double-click the file named **`run_app.bat`**.
   - *First run:* It will automatically build `.venv`, install all packages, and launch the GUI.
   - *Subsequent runs:* It will launch the GUI immediately.

> **Note:** If `run_app.bat` closes or crashes, check `crash_log.txt` in the root folder for diagnostic details.

---

### Method 2: Conda Environment Setup (Cross-Platform: Windows, macOS, Linux)

Recommended for Anaconda or Miniconda users.

#### 1. Open Terminal or Anaconda Prompt and create the environment:
```bash
conda create -n f3dca_env python=3.10 tk -y
```

#### 2. Activate the environment:
```bash
conda activate f3dca_env
```

#### 3. Install required dependencies:
```bash
pip install numpy pandas openpyxl scipy scikit-learn matplotlib pillow tf-keras tensorflow
```

#### 4. Launch the application:
```bash
python ClusteringApp.py
```
*(Replace `ClusteringApp.py` with your script's filename if different).*

---

### Method 3: Standard Python Virtual Environment (`venv` + `pip`)

Works on any system with Python 3.10+ installed.

#### 1. Open Terminal (macOS/Linux) or Command Prompt (Windows):
Navigate to the extracted directory:
```bash
cd path/to/F3DCA-directory
```

#### 2. Create the virtual environment:
- **Windows:**
  ```cmd
  python -m venv .venv
  call .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 3. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install numpy pandas openpyxl scipy scikit-learn matplotlib pillow tf-keras tensorflow
```

*(Optional acceleration)*:
```bash
pip install numba
```

#### 4. Run the program:
```bash
python ClusteringApp.py
```

---

## Linux / Ubuntu Note (Tkinter Dependency)

If you are running on Ubuntu or Debian, Tkinter must be installed at the operating-system level:
```bash
sudo apt-get update
sudo apt-get install python3-tk
```

---

## Troubleshooting & Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'pandas'` | Virtual environment was interrupted during setup | Delete the `.venv` folder and run `run_app.bat` again (or rerun Method 2 / 3). |
| `Could not find a version that satisfies tensorflow==2.15.0` | You are running Python 3.12 | Install without pinning version (`pip install tensorflow tf-keras`), as handled in the commands above. |
| Window crashes without errors | Keras 3 compatibility issue with TF 2.16+ | Set environment variable `TF_USE_LEGACY_KERAS=1` (already included in `run_app.bat`). |
| Missing Excel reader engine | `openpyxl` is missing | Run `pip install openpyxl`. |

---

## Input Data Format Requirements

Input files must be Microsoft Excel spreadsheets (`.xlsx`):
1. **First 3 Columns:** Must contain Cartesian coordinates named or ordered as **`X`**, **`Y`**, **`Z`**.
2. **Sheet Structure:** Each sheet within an `.xlsx` file is processed as an unbroken, continuous temporal trajectory recording.
3. Multiple `.xlsx` files can be loaded simultaneously to represent distinct experimental or biological groups.