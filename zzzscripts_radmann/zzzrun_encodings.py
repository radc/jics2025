import subprocess
import os
import shutil
from itertools import product
from zzzzproperties import *

for q in range(1, 10):
    for a, w in product(bits, repeat=2):
        outPath = f"{encodingsPath}/a{a}_w{w}/{q}"

        if os.path.exists(outPath):
            jsonFiles = [f for f in os.listdir(outPath) if f.endswith(".json")]
            if len(jsonFiles) == 6:
                print(f"Skipping a{a}_w{w}/{q} - DONE")
                continue
            else:
                print(f"Incomplete or corrupted a{a}_w{w}/{q} - Removing and running again")
                shutil.rmtree(outPath)

        print(f"\nRunning a{a}_w{w}/{q}\n")
        try:
            subprocess.run([
                "python", "ztest_video.py",
                "-m", "ssf2020",
                "-d", "/data/Datasets/vimeo_septuplet",
                "--quality", f"{q}",
                "--cuda",
                "--split-list", f"calibrate{sequences}",
                "--num-frames", "7",
                "--patch-size", f"{patchSize[0]}", f"{patchSize[1]}",
                "--aimet-calibrate",
                "--aimet-path-encodings", outPath,
                "--aimet-activation-bw", f"{a}",
                "--aimet-weight-bw", f"{w}",
            ], check=True)

        except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
            print(f"Error running subprocess for a{a}_w{w}/{q}: {e}")
