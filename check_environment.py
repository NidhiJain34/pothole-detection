import sys
import numpy
from PIL import Image

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("Pillow:", Image.__version__ if hasattr(Image, "__version__") else "Installed")

print("\nEnvironment verification successful!")