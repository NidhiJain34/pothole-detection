import subprocess
import sys
from pathlib import Path

# Dataset folder
project_folder = Path(__file__).resolve().parent

# Create a separate virtual environment
venv_folder = project_folder / ".venv"

if not venv_folder.exists():
    print("Creating virtual environment...")

    result = subprocess.run([
        sys.executable,
        "-m",
        "venv",
        str(venv_folder)
    ])

    if result.returncode != 0:
        print("Could not create the virtual environment.")
        print("Please send me the error shown above.")
        raise SystemExit(1)
else:
    print("Virtual environment already exists.")

# Locate pip inside the virtual environment
if sys.platform == "win32":
    python_executable = venv_folder / "Scripts" / "python.exe"
else:
    python_executable = venv_folder / "bin" / "python"

print("\nInstalling Pillow and NumPy...")

result = subprocess.run([
    str(python_executable),
    "-m",
    "pip",
    "install",
    "pillow",
    "numpy"
])

if result.returncode == 0:
    print("\nSUCCESS: Pillow and NumPy are installed.")
    print("Virtual environment:", venv_folder)
else:
    print("\nInstallation failed. Send me the error above.")