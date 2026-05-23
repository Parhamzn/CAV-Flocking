"""Path bootstrap — import this at the top of every experiment script.

Adds the project root (the parent of this `experiments/` folder) to
sys.path so that `from flocking_lib.X import Y` works regardless of
whether the script is launched from the project root, from inside
`experiments/`, or via `python -m`.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
