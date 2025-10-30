import re
import os

def getExpectedSequences(folder):
    testListPath = f"/data/Radmann/{folder}/test.list"
    if not os.path.exists(testListPath):
        return set()
    with open(testListPath, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def testLogBlockCompleteness(folder, block):
    expectedSeqs = getExpectedSequences(folder)
    foundSeqs = set()
    for line in block.splitlines():
        match = re.search(r"Seq ([^ ]+)", line)
        if match:
            seqName = match.group(1)
            if seqName.endswith(".yuv"):
                seqName = seqName[:-4]
            foundSeqs.add(seqName)

    print(f"Expected ({len(expectedSeqs)}): {sorted(expectedSeqs)}")
    print(f"Found    ({len(foundSeqs)}): {sorted(foundSeqs)}")
    print(f"Missing: {sorted(expectedSeqs - foundSeqs)}")
    print(f"Extra:   {sorted(foundSeqs - expectedSeqs)}")
    print("Is complete?", expectedSeqs.issubset(foundSeqs))

# Exemplo de bloco, substitua pelo seu real
block = """
[Epoch 0] Seq BasketballDrive_1920x1080_50.yuv (#000) → Loss: 0.3836 ...
[Epoch 0] Seq BQTerrace_1920x1080_60.yuv (#001) → Loss: 1.1747 ...
[Epoch 0] Seq Cactus_1920x1080_50.yuv (#002) → Loss: 0.8738 ...
[Epoch 0] Seq Kimono1_1920x1080_24.yuv (#003) → Loss: 0.2901 ...
[Epoch 0] Seq ParkScene_1920x1080_24.yuv (#004) → Loss: 0.5676 ...
"""

testLogBlockCompleteness("HEVC-B", block)
