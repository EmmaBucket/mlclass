"""
Make every entry point run in the ONE Python that has the project's libraries.

The project lives in the `mlclass` conda environment (OpenCV, MediaPipe,
Selenium, scikit-learn). The Mac also has a plain Python 3.13 and a Python 3.14
from python.org, and VS Code's "Run" button happily picks one of those -- which
then fails at the first import with an error that looks like the project is
broken. It is not; it is just the wrong interpreter.

Import this at the top of any script meant to be run directly. If we are not in
the right environment and the right one exists, the script re-launches itself
there. If it does not exist, we say exactly what to install.
"""
import os
import sys

MLCLASS = "/opt/anaconda3/envs/mlclass/bin/python"


def ensure():
    here = os.path.realpath(sys.executable)
    if "/envs/mlclass/" in here:
        return                                   # already right
    if os.path.exists(MLCLASS):
        script = os.path.abspath(sys.argv[0])
        print(f"(switching to the project environment: {MLCLASS})")
        os.execv(MLCLASS, [MLCLASS, script] + sys.argv[1:])
    sys.stderr.write(
        "\nThis project needs its conda environment, which was not found at\n"
        f"  {MLCLASS}\n"
        "Create it once with:\n"
        "  conda create -n mlclass python=3.10 && conda activate mlclass && "
        "pip install -r requirements.txt\n\n")
    sys.exit(2)


ensure()
