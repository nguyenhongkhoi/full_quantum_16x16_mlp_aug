"""Run Mammoth's main.py with a log folder, an early stop and a resume. Put next to main.py and run it from there.

    python run_mammoth.py --name NAME [--end_at N] [--resume CHECKPOINT] <every main.py argument>

--name      The log goes to logs/log_NAME.txt, with both stdout and stderr, so the Forgetting lines are in it too.
            The file is only appended to, never overwritten. Progress bars keep only their last state.
--end_at N  Stop after task N, counted from 1, and skip the final test, so the test set stays unused. Backward
            Transfer and Forgetting are still printed, over the tasks done. Without --end_at the run goes to the end
            and runs the final test as before.
--resume    Disabled (2026-11-07): on rot-mnist Mammoth rebuilds the earlier tasks with other angles, and it reuses
            the saved stop_after. Run from the start instead.

Checkpoints are always saved after every task (--savecheck task). Each task's angle is printed as [ROTATION].
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import random
import re
import runpy
import sys
from pathlib import Path


def _last_state(text: str) -> str:
    """A progress bar redraws itself after a carriage return; keep only what is left on screen."""
    text = text.rstrip("\r")
    return text[text.rfind("\r") + 1:]


class Tee:
    """Writes to the console unchanged and to the log file line by line."""

    def __init__(self, stream, handle):
        self.stream, self.handle, self.pending = stream, handle, ""

    def write(self, text: str) -> int:
        self.stream.write(text)
        *lines, rest = (self.pending + text).split("\n")
        for line in lines:
            self.handle.write(_last_state(line) + "\n")
        self.pending = _last_state(rest) if "\r" in rest else rest
        self.handle.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main() -> None:
    if not (Path("main.py").is_file() and Path("utils").is_dir()):
        sys.exit("run_mammoth.py must run inside the Mammoth folder (where main.py is). In mammoth_full_package: run "
                 "'powershell -ExecutionPolicy Bypass -File .\\setup.ps1' once, then 'cd .\\mammoth' (or use "
                 "run_all.ps1 -Only <backbone>). mammoth_files\\ only holds the files that setup copies into Mammoth.")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="log name: logs/log_NAME.txt")
    parser.add_argument("--end_at", type=int, default=None, help="stop after this task, counted from 1")
    parser.add_argument("--resume", default=None, help="checkpoint .pt of an earlier run_mammoth.py run")
    ours, rest = parser.parse_known_args()
    for flag in ("--stop_after", "--start_from", "--loadcheck"):
        if flag in rest:
            parser.error(f"{flag} is set by run_mammoth.py; use --end_at and --resume instead")
    if "--savecheck" in rest:
        if rest[rest.index("--savecheck") + 1] != "task":
            parser.error("run_mammoth.py needs --savecheck task")
    else:
        rest += ["--savecheck", "task"]

    start = 0
    if ours.resume is not None:
        # Disabled 2026-11-07. Mammoth rebuilds the earlier tasks before loading, and on rot-mnist they get new
        # angles (training also draws from np.random), so old tasks are evaluated on angles never trained on.
        # main.py also takes the checkpoint's args as defaults, so a saved stop_after ends the run at once.
        parser.error("--resume is disabled: on rot-mnist the earlier tasks get different angles. Run from the start.")
        match = re.search(r"_(\d+)\.pt$", ours.resume)
        if match is None or not Path(ours.resume).is_file():
            parser.error("--resume must be an existing checkpoint named <name>_<task>.pt")
        start = int(match.group(1)) + 1
        rest += ["--loadcheck", ours.resume, "--start_from", str(start)]
    if ours.end_at is not None:
        if ours.end_at <= start:
            parser.error(f"--end_at {ours.end_at} is not after the resumed task {start}")
        rest += ["--stop_after", str(ours.end_at)]

    Path("logs").mkdir(exist_ok=True)
    handle = open(Path("logs") / f"log_{ours.name}.txt", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, handle), Tee(sys.stderr, handle)
    print(f"===== run_mammoth.py {datetime.datetime.now():%Y-%m-%d %H:%M:%S}: {' '.join(sys.argv[1:])}", flush=True)
    print(f"===== main.py {' '.join(rest)}", flush=True)

    # Imported after the Tee, so that Mammoth's log handler writes through it.
    import numpy as np
    import torch
    import utils.training as training
    from datasets.transforms.rotation import Rotation
    from utils.conf import get_checkpoint_path

    original_rotation = Rotation.__init__

    def logged_rotation(self, *args, **kwargs):
        original_rotation(self, *args, **kwargs)
        print(f"[ROTATION] {self.degrees:.4f}", flush=True)

    Rotation.__init__ = logged_rotation

    original_save = training.save_mammoth_checkpoint

    def save(task, end_task, args, *more, **kwargs):
        original_save(task, end_task, args, *more, **kwargs)
        name = os.path.join(get_checkpoint_path(), f"{args.ckpt_name}_{task}")
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   name + "_rng.pt")
        if ours.end_at is not None and task + 1 == ours.end_at:
            # The final test only runs when args.validation is set; it is checked after this last save.
            args.validation = None
            logging.info(f"run_mammoth.py: stopped after task {ours.end_at}, final test skipped. "
                         f"Resume with --resume {name}.pt")

    training.save_mammoth_checkpoint = save

    original_load = training.mammoth_load_checkpoint

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        states = ours.resume[:-len(".pt")] + "_rng.pt"
        if os.path.isfile(states):
            saved = torch.load(states, map_location="cpu", weights_only=False)
            random.setstate(saved["python"])
            np.random.set_state(saved["numpy"])
            torch.set_rng_state(saved["torch"])
            if saved["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved["cuda"])
            logging.info(f"run_mammoth.py: random states restored from {states}")
        else:
            logging.warning(f"run_mammoth.py: {states} not found. The run resumes, but the rotation angles of the "
                            "next tasks will differ from an uninterrupted run.")
        return result

    training.mammoth_load_checkpoint = load

    sys.argv = ["main.py"] + rest
    try:
        runpy.run_path("main.py", run_name="__main__")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
