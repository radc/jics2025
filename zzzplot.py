import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from scipy import interpolate

logFiles = {
    "HEVC-B": "NEW_HEVC-B_32",
    "UVG": "NEW_UVG_32"
}

headerPattern = re.compile(r"\[OK\] (HEVC-B|UVG) (\w+|anchor) (\d+):")
metricPattern = re.compile(r"PSNR: (-?[\d.]+) dB, BPP: ([\d.]+)")

data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# --- Read logs and aggregate data ---
for dataset, filepath in logFiles.items():
    if not os.path.exists(filepath):
        print(f"[WARN] Log file {filepath} not found.")
        continue

    with open(filepath, 'r') as f:
        lines = f.readlines()

    currentConfig = None
    currentQuality = None

    for line in lines:
        headerMatch = headerPattern.match(line)
        if headerMatch:
            _, currentConfig, q = headerMatch.groups()
            currentQuality = int(q)
            continue

        metricMatch = metricPattern.search(line)
        if metricMatch and currentConfig is not None:
            psnr, bpp = map(float, metricMatch.groups())
            data[dataset][currentConfig][currentQuality].append((psnr, bpp))

# --- Helper function to average values ---
def average(vals):
    return sum(vals) / len(vals) if vals else 0

# --- Prepare data for plotting (mean per quality) ---
def prepareRdData(datasetData):
    rdData = {}
    for config, qualDict in datasetData.items():
        psnrList = []
        bppList = []
        for q in qualDict.keys():
            values = qualDict[q]
            if values:
                avgPsnr = average([v[0] for v in values])
                avgBpp = average([v[1] for v in values])
                psnrList.append(avgPsnr)
                bppList.append(avgBpp)
        rdData[config] = (np.array(bppList), np.array(psnrList))
    return rdData

# --- BD-Rate calculation from https://tech.almalinux.org/bdrate.py ---
def bdRate(rdRef, rdTest):
    xRef, yRef = rdRef
    xTest, yTest = rdTest

    xRef, uniqueIndicesRef = np.unique(xRef, return_index=True)
    yRef = yRef[uniqueIndicesRef]

    xTest, uniqueIndicesTest = np.unique(xTest, return_index=True)
    yTest = yTest[uniqueIndicesTest]

    logXRef = np.log(xRef)
    logXTest = np.log(xTest)

    if len(logXRef) < 3 or len(logXTest) < 3:
        return None

    pRef = interpolate.PchipInterpolator(logXRef, yRef)
    pTest = interpolate.PchipInterpolator(logXTest, yTest)

    low = max(logXRef.min(), logXTest.min())
    high = min(logXRef.max(), logXTest.max())

    if low >= high:
        return None

    intRef = pRef.integrate(low, high)
    intTest = pTest.integrate(low, high)
    avgDiff = (intTest - intRef) / (high - low)

    bdRatePercent = (np.exp(avgDiff) - 1) * 100
    return bdRatePercent


# --- Plot RD curves ---
def plotRdCurves(title, rdData, filename):
    plt.figure(figsize=(12,7))
    allBpps = []
    allPsnrs = []
    for config, (bpps, psnrs) in sorted(rdData.items()):
        if len(bpps) >= 3:
            plt.plot(bpps, psnrs, marker='o', label=config, markersize=5)
            allBpps.extend(bpps)
            allPsnrs.extend(psnrs)

    if allBpps and allPsnrs:
        bppMin, bppMax = min(allBpps), max(allBpps)
        psnrMin, psnrMax = min(allPsnrs), max(allPsnrs)
        marginBpp = (bppMax - bppMin) * 0.05
        marginPsnr = (psnrMax - psnrMin) * 0.05
        plt.xlim(bppMin - marginBpp, bppMax + marginBpp)
        plt.ylim(psnrMin - marginPsnr, psnrMax + marginPsnr)

    plt.title(f"Rate-Distortion Curves - {title}")
    plt.xlabel("Average BPP")
    plt.ylabel("Average PSNR (dB)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

def comboSortKey(combo):
    if combo == "anchor":
        return (-1, -1)
    parts = combo.split("_")
    aNum = int(parts[0][1:])
    wNum = int(parts[1][1:])
    return (aNum, wNum)

# --- Calculate and print BD-rate table ---
def printBdRateTable(rdData, reference="anchor", datasetName="Dataset"):
    print(f"\nBD-Rate comparison to reference: {reference} - {datasetName}")
    print(f"{'Config':<15} | {'BD-Rate (%)':>10}")
    print("-" * 28)
    refData = rdData.get(reference)
    if refData is None:
        print("Reference config not found.")
        return

    combos = list(rdData.keys())
    if "a8_w8" in combos:
        combos.remove("a8_w8")
        combos = ["a8_w8"] + sorted(combos, key=comboSortKey)
    else:
        combos = sorted(combos, key=comboSortKey)

    for config in combos:
        if config == reference:
            continue
        bd = bdRate(refData, rdData[config])
        if bd is None:
            print(f"{config:<15} | {'N/A':>10}")
        else:
            print(f"{config:<15} | {bd:10.4f}")


if __name__ == "__main__":
    # Prepare RD data per dataset
    for dataset in data:
        rdCurves = prepareRdData(data[dataset])
        plotRdCurves(dataset, rdCurves, f"{dataset}.png")
        printBdRateTable(rdCurves, reference="anchor", datasetName=dataset)

    # Prepare general RD data (merged)
    mergedData = defaultdict(lambda: defaultdict(list))
    for datasetData in data.values():
        for config, qDict in datasetData.items():
            for q, vals in qDict.items():
                mergedData[config][q].extend(vals)

    mergedRdCurves = prepareRdData(mergedData)
    plotRdCurves("Overall", mergedRdCurves, "Overall.png")
    printBdRateTable(mergedRdCurves, reference="anchor", datasetName="Overall")
