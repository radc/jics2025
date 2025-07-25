import subprocess
import re
from itertools import product
import os

folders = [
    "HEVC-B",
    "UVG"
]

frames = 32

def log(folder, message):
    lines = message.splitlines()
    cleanedLines = []

    for line in lines:
        if re.search(r"Unsupported op type \w+", line):
            continue
        if "Selecting DefaultOpInstanceConfigGenerator" in line:
            continue
        if line.startswith("Warning: no checkpoint provided"):
            continue

        cleanedLines.append(line)

    cleanedMessage = "\n".join(cleanedLines)

    with open(f"{folder}_{frames}_NEW_1", "a", encoding="utf-8") as f:
        f.write(cleanedMessage + "\n")

def loadCompletedKeys(folder):
    if not os.path.exists(folder):
        return set()

    completed = set()
    
    with open(folder, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\[OK\] (.+?) (a\d+_w\d+|anchor) (\d+):", line.strip())

            if match:
                folder, mode, q = match.groups()
                key = f"{folder} {mode} {q}"
                completed.add(key)
                
    return completed

for folder in folders:
    completedKeys = loadCompletedKeys(folder)

    for q in range(1, 10):
        anchorKey = f"{folder} anchor {q}"

        if anchorKey in completedKeys:
            print(f"[SKIP] {anchorKey}")

        else:
            cmd = [
                "python", "ztest_video.py",
                "-m", "ssf2020",
                "-d", f"/data/Radmann/{folder}",
                "--patch-size", "1920", "1024",
                "--cuda",
                "--num-frames", f"{frames}",
                "--quality", f"{q}",
                "--test-batch-size", "2",
                "--ignore-sequence-folder"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                log(folder, f"\n[OK] {folder} anchor {q}: \n\n{result.stdout}")

            except subprocess.CalledProcessError as e:
                log(folder, f"[ERROR] {folder} anchor {q}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")

            key = f"{folder} a1_w1 {q}"

            if key in completedKeys:
                print(f"[SKIP] {key}")
                continue

            encodingPath = f"encodings/a1_w1/{q}"

            cmd = [
                "python", "ztest_video.py",
                "-m", "ssf2020",
                "-d", f"/data/Radmann/{folder}",
                "--patch-size", "1920", "1024",
                "--cuda",
                "--num-frames", f"{frames}",
                "--quality", f"{q}",
                "--test-batch-size", "2",
                "--aimet-load-encodings",
                "--aimet-path-encodings", encodingPath,
                "--ignore-sequence-folder"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                log(folder, f"\n[OK] {folder} a1_w1 {q}: \n\n{result.stdout}")

            except subprocess.CalledProcessError as e:
                log(folder, f"[ERROR] {folder} a1_w1 {q}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
