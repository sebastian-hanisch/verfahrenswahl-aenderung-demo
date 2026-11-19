"""Gemeinsame Einrichtung: der Demo-Ordner liegt auf dem Suchpfad; die Rechnungen laufen mit einem Thread (reproduzierbare Rechenzeiten)."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
