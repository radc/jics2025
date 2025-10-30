import re
import os
from zzzzproperties import *
import matplotlib.pyplot as plt
from matplotlib import cycler
from collections import defaultdict
import numpy as np
import bjontegaard as bd

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

def average(vals):
    return sum(vals) / len(vals) if vals else 0

def prepareRdData(datasetData):
    rdData = {}
    for config, qualDict in datasetData.items():
        psnrList = []
        bppList = []
        for q in sorted(qualDict.keys()):
            values = qualDict[q]
            if values:
                avgPsnr = average([v[0] for v in values])
                avgBpp = average([v[1] for v in values])
                psnrList.append(avgPsnr)
                bppList.append(avgBpp)
                
        rdData[config] = (np.array(bppList), np.array(psnrList))

    return rdData

def bdRate(rdRef, rdTest):
    xRef, yRef = rdRef
    xTest, yTest = rdTest

    xRef, uniqueIndicesRef = np.unique(xRef, return_index=True)
    yRef = yRef[uniqueIndicesRef]

    xTest, uniqueIndicesTest = np.unique(xTest, return_index=True)
    yTest = yTest[uniqueIndicesTest]

    try:
        bdRateValue = bd.bd_rate(xRef, yRef, xTest, yTest, method='akima', min_overlap=0.3)
        bdPsnrValue = bd.bd_psnr(xRef, yRef, xTest, yTest, method='akima', min_overlap=0.3)
        
        return bdRateValue, bdPsnrValue
    
    except Exception:
        return None

def plotRdCurves(title, rdData, filename, reference="anchor"):
    plt.figure(figsize=(12,7))
    allBpps = []
    allPsnrs = []

    for config, (bpps, psnrs) in sorted(rdData.items(), key=lambda x: comboSortKey(x[0])):
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
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fontsize='small', frameon=True)


    refData = rdData.get(reference)
    if refData is not None:
        rows = []
        for config in sorted(rdData.keys(), key=comboSortKey):
            if config == reference:
                continue
            bdResult = bdRate(refData, rdData[config])
            if bdResult is not None:
                bdRateValue, _ = bdResult
                rows.append([config, f"{bdRateValue:+.4f} %"])

        if rows:
            colLabels = ["Config", "BD-Rate (%)"]
            plt.table(cellText=rows, colLabels=colLabels, cellLoc='left', colLoc='center',
                      loc='lower right', bbox=[0.76, 0.04, 0.22, 0.35],
                      edges='closed')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

def comboSortKey(combo):
    if combo == "anchor":
        return (-1, -1)
    m = re.match(r'a(\d+)_w(\d+)', combo)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    else:
        return (float('inf'), float('inf'))

def printBdRateTable(rdData, reference="anchor", datasetName="Dataset"):
    print(f"\nBD-Rate comparison to reference: {reference} - {datasetName}")
    print(f"{'Config':<10} | {'BD-Rate (%)':>10}")
    print("-" * 28)
    refData = rdData.get(reference)
    if refData is None:
        print("Reference config not found.")
        return

    combos = sorted(rdData.keys(), key=comboSortKey)

    for config in combos:
        if config == reference:
            continue
        bd = bdRate(refData, rdData[config])
        if bd is None:
            print(f"{config:<10} | {'N/A':>10}")
        else:
            bdRateValue, _ = bd
            print(f"{config:<10} | {bdRateValue:10.4f}")

def findMostQuantized(rdData):
    minBppAvg = float("inf")
    worstConfig = None
    for config, (bpps, _) in rdData.items():
        if len(bpps) >= 3:
            avgBpp = np.mean(bpps)
            if avgBpp < minBppAvg:
                minBppAvg = avgBpp
                worstConfig = config
    return worstConfig

if __name__ == "__main__":
    for dataset in data:
        rdCurves = prepareRdData(data[dataset])
        ref = findMostQuantized(rdCurves)
        plotRdCurves(dataset, rdCurves, f"{dataset}.png", reference=ref)
        printBdRateTable(rdCurves, reference=ref, datasetName=dataset)

    mergedData = defaultdict(lambda: defaultdict(list))
    for datasetData in data.values():
        for config, qDict in datasetData.items():
            for q, vals in qDict.items():
                mergedData[config][q].extend(vals)

    mergedRdCurves = prepareRdData(mergedData)
    autoReference = findMostQuantized(mergedRdCurves)
    plotRdCurves("Overall", mergedRdCurves, "Overall.png", reference=autoReference)
    printBdRateTable(mergedRdCurves, reference=autoReference, datasetName="Overall")
