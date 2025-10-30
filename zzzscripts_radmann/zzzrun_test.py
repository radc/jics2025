import os
import re
import subprocess
from itertools import product
from zzzzproperties import *

combos = ["anchor"] + [f"a{a}_w{w}" for a in bits for w in bits]
qualities = list(range(1, 10))
saveDir = "sheets"

def getExpectedSequences(folder):
    testListPath = f"/data/Radmann/{folder}/test.list"
    if not os.path.exists(testListPath):
        return set()
    with open(testListPath, "r", encoding="utf-8") as f:
        return set(os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip())

def cleanMessage(message):
    message = re.sub(
        r"^Warning: no checkpoint provided; using pretrained weights only\.\n?",
        "", message, flags=re.MULTILINE,
    )
    message = re.sub(r"^\[.*?INFO.*?\].*\n?", "", message, flags=re.MULTILINE)
    message = re.sub(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - Quant - INFO - (Unsupported op type .+|Selecting DefaultOpInstanceConfigGenerator .+)$\n?",
        "", message, flags=re.MULTILINE,
    )
    message = re.sub(r"\n{2,}", "\n\n", message).strip()
    return message

def extractLogBlock(path, combo, quality):
    header = f"[OK] {os.path.basename(path).replace(f'_{patchSize[0]}_{frames}_{sequences}', '')} {combo} {quality}:"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    for i, block in enumerate(blocks):
        if block.startswith(header):
            return block, i, blocks
    return None, None, blocks

def isLogBlockComplete(block, expectedSeqs):
    foundSeqs = set()
    for line in block.splitlines():
        match = re.search(r"Seq ([^ ]+)", line)
        if match:
            seqName = match.group(1).removesuffix(".yuv")
            foundSeqs.add(seqName)
    return expectedSeqs.issubset(foundSeqs)

def replaceLogBlock(path, blockIndex, newBlock, blocks):
    blocks[blockIndex] = newBlock.strip()
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks).strip() + "\n\n")

def appendLogBlock(path, newBlock):
    with open(path, "a", encoding="utf-8") as f:
        f.write(newBlock.strip() + "\n\n")

def run(folder, combo, quality):
    path = f"{folder}_{patchSize[0]}_{frames}_{sequences}"
    expectedSeqs = getExpectedSequences(folder)
    block, blockIndex, blocks = extractLogBlock(path, combo, quality)

    if block:
        if isLogBlockComplete(block, expectedSeqs):
            print(f"[OK] {folder} {combo} {quality} done.")
            return
        else:
            print(f"[WARN] {folder} {combo} {quality} incomplete.")
    else:
        blockIndex = None

    if combo == "anchor":
        cmd = [
            "python", "ztest_video.py",
            "-m", "ssf2020",
            "-d", f"/data/Radmann/{folder}",
            "--patch-size", "1920", "1024",
            "--cuda",
            "--num-frames", str(frames),
            "--quality", str(quality),
            "--test-batch-size", "2",
            "--ignore-sequence-folder",
        ]
    else:
        encodingPath = f"{encodingsPath}/{combo}/{quality}"
        cmd = [
            "python", "ztest_video.py",
            "-m", "ssf2020",
            "-d", f"/data/Radmann/{folder}",
            "--patch-size", "1920", "1024",
            "--cuda",
            "--num-frames", str(frames),
            "--quality", str(quality),
            "--test-batch-size", "2",
            "--aimet-load-encodings",
            "--aimet-path-encodings", encodingPath,
            "--ignore-sequence-folder",
        ]

    print(f"[RUN] {folder} {combo} {quality}", end="\r")
    result = subprocess.run(cmd, capture_output=True, text=True)
    cleaned = cleanMessage(result.stdout)
    newBlock = f"[OK] {folder} {combo} {quality}:\n\n{cleaned}"

    if blockIndex is not None:
        replaceLogBlock(path, blockIndex, newBlock, blocks)
        print(f"[RERUN] {folder} {combo} {quality} done.")
    else:
        appendLogBlock(path, newBlock)
        print(f"[RUN] {folder} {combo} {quality} done.")

def processAll():
    for folder in folders:
        path = f"{folder}_{patchSize[0]}_{frames}_{sequences}"
        if not os.path.exists(path):
            open(path, "w").close()

        for quality in qualities:
            run(folder, "anchor", quality)
            for a, w in product(bits, bits):
                combo = f"a{a}_w{w}"
                run(folder, combo, quality)

if __name__ == "__main__":
    processAll()
