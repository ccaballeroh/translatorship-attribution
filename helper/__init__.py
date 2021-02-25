__all__ = ["analysis", "preprocessing", "features", "utils", "IN_COLAB", "ROOT"]
__version__ = "1.1"
__author__ = "Christian Caballero <cdch10@gmail.com>"


import sys
from pathlib import Path

# flag to change root folder if running in colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path(r"./drive/My Drive/translator-attribution/")  # Root folder in colab
    print("In colab!")
else:
    ROOT = Path(r".")  # Root folder if running locally

