"""Turn Mammoth terminal logs (saved with Tee-Object) into CSV files for plotting. Standard library only.

Run in Mammoth's folder: python export_mammoth_logs.py [log files]. With no files, reads every log_*.txt in the
folder and in logs/, except log_diagnose_*. Writes mammoth_mean.csv (run, after_task, mean) and mammoth_tasks.csv
(run, after_task, task, accuracy). Only "[INFO]" lines are read, so each value appears once.
The "final" line is the real test set: it is kept with after_task = final and must not be used to choose.
"""
import csv
import re
import sys
from pathlib import Path

MEAN = re.compile(r"\[INFO\].*Accuracy for (\d+|final) task\(s\): \[Domain-IL\]: ([\d.]+) %")
RAW = re.compile(r"\[INFO\].*Raw accuracy values: Domain-IL \[([^\]]*)\]")


def read_text(path: Path) -> str:
    """PowerShell 5.1 Tee-Object writes UTF-16; other tools write UTF-8."""
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8", errors="replace")


def main() -> None:
    paths = [Path(name) for name in sys.argv[1:]] or sorted(
        path for folder in (Path("."), Path("logs")) for path in folder.glob("log_*.txt")
        if not path.name.startswith("log_diagnose"))
    means, tasks = [], []
    for path in paths:
        run = path.stem.removeprefix("log_")
        after_task = None
        for line in read_text(path).splitlines():
            if match := MEAN.search(line):
                after_task = match.group(1)
                means.append((run, after_task, float(match.group(2))))
            elif (match := RAW.search(line)) and after_task is not None:
                values = [float(value) for value in match.group(1).split(",") if value.strip()]
                tasks.extend((run, after_task, task, value) for task, value in enumerate(values, start=1))
        print(f"{path.name}: {sum(1 for row in means if row[0] == run)} mean lines")
    with open("mammoth_mean.csv", "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([("run", "after_task", "mean"), *means])
    with open("mammoth_tasks.csv", "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([("run", "after_task", "task", "accuracy"), *tasks])
    print("Wrote mammoth_mean.csv and mammoth_tasks.csv")


if __name__ == "__main__":
    main()
