"""Run Mammoth's main.py with the same arguments and print each task's rotation. Put next to main.py."""
import runpy
import sys

from datasets.transforms.rotation import Rotation

original_init = Rotation.__init__


def logged_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    print(f"ROTATION {self.degrees:.2f}", flush=True)


Rotation.__init__ = logged_init
sys.argv[0] = "main.py"
runpy.run_path("main.py", run_name="__main__")
