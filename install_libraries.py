import subprocess
import sys

libraries = ["pillow", "numpy"]

for library in libraries:
    print(f"\nInstalling {library}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", library],
        text=True
    )

    if result.returncode == 0:
        print(f"{library} installed successfully.")
    else:
        print(f"Installation of {library} failed.")
        print("Please check the error above.")
        break

print("\nInstallation attempt finished.")