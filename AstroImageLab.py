#build with: pyinstaller AstroImageLab.spec
#build with full recompile: pyinstaller --clean AstroImageLab.spec

#Create a conda environment and install dependencies:
# conda create -n astrolab_build python=3.12
# conda activate astrolab_build
# conda install -c conda-forge pyqt6 astropy photutils scipy numpy matplotlib astroalign pillow pywavelets
# pip install xisf xhtml2pdf

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Astro Image Lab")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
