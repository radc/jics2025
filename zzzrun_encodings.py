import subprocess
import os
from itertools import product

for q in range(1, 10):

    if(os.path.exists(f"encodings/a1_w1/{q}")):
        print(f"Skipping encoding a1_w1/{q} - DONE")

    else:
        try:
            subprocess.run([
                "python", "ztest_video.py",
                "-m", "ssf2020",
                "-d", "/data/Datasets/vimeo_septuplet",
                "--quality", f"{q}",
                "--cuda",
                "--aimet-calibrate",
                "--aimet-path-encodings", f"encodings/a1_w1/{q}",
                "--aimet-activation-bw", "1",
                "--aimet-weight-bw", "1",
            ])
            
        except (subprocess.SubprocessError, subprocess.CalledProcessError) as e:
            print(e.with_traceback)
        