import re
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import bjontegaard as bd
import pandas as pd

filePattern = re.compile(r'^(HEVC-B|UVG)_(\d+)_(\d+)_(\d+)$')
headerPattern = re.compile(r"\[OK\] (HEVC-B|UVG) (\w+|anchor) (\d*[13579]):")
metricPattern = re.compile(r"PSNR: (-?[\d.]+) dB, BPP: ([\d.]+)")

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

def curvesOverlap(x1, x2):
    return not (max(x1) < min(x2) or max(x2) < min(x1))

def bdRate(rdRef, rdTest):
    xRef, yRef = rdRef
    xTest, yTest = rdTest

    xRef, uniqueIndicesRef = np.unique(xRef, return_index=True)
    yRef = yRef[uniqueIndicesRef]

    xTest, uniqueIndicesTest = np.unique(xTest, return_index=True)
    yTest = yTest[uniqueIndicesTest]

    if not curvesOverlap(xRef, xTest):
        return None

    try:
        bdRateValue = bd.bd_rate(xRef, yRef, xTest, yTest, method='akima', min_overlap=0)
        bdPsnrValue = bd.bd_psnr(xRef, yRef, xTest, yTest, method='akima', min_overlap=0)
        return bdRateValue, bdPsnrValue
    except Exception:
        return None

def comboSortKey(combo):
    if combo == "anchor":
        return (-1, -1)
    m = re.match(r'a(\d+)_w(\d+)', combo)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    else:
        return (float('inf'), float('inf'))

# ---------------- Overall ponto a ponto ----------------
def build_overall_curves_pointwise(rd_hevc, rd_uvg):
    overall = {}
    all_configs = set(rd_hevc.keys()) | set(rd_uvg.keys())
    for config in all_configs:
        bppA, psnrA = rd_hevc.get(config, (None, None))
        bppB, psnrB = rd_uvg.get(config, (None, None))

        if bppA is None and bppB is None:
            continue
        if bppA is None:
            overall[config] = (np.array(bppB, copy=True), np.array(psnrB, copy=True))
            continue
        if bppB is None:
            overall[config] = (np.array(bppA, copy=True), np.array(psnrA, copy=True))
            continue

        # Mesma quantidade de pontos (ponto a ponto)
        min_len = min(len(bppA), len(bppB))
        bpp_overall = (bppA[:min_len] + bppB[:min_len]) / 2
        psnr_overall = (psnrA[:min_len] + psnrB[:min_len]) / 2

        overall[config] = (bpp_overall, psnr_overall)
    return overall

def compute_overall_bdrates(bdr_hevc, bdr_uvg):
    overall = {}
    all_cfgs = set(bdr_hevc.keys()) | set(bdr_uvg.keys())
    for cfg in all_cfgs:
        vals = []
        if cfg in bdr_hevc:
            vals.append(bdr_hevc[cfg])
        if cfg in bdr_uvg:
            vals.append(bdr_uvg[cfg])
        if vals:
            overall[cfg] = sum(vals) / len(vals)
    return overall

# ---------------- Plots e Tabelas ----------------
def plotRdCurves(title, rdData, filename, reference="anchor", bdRateTable=None):
    plt.figure(figsize=(12, 7))
    allBpps = []
    allPsnrs = []

    for config, (bpps, psnrs) in sorted(rdData.items(), key=lambda x: comboSortKey(x[0])):
        if len(bpps) >= 1:
            plt.plot(bpps, psnrs, marker='o', label=config, markersize=5)
            allBpps.extend(bpps)
            allPsnrs.extend(psnrs)

    if allBpps and allPsnrs:
        bppMin, bppMax = min(allBpps), max(allBpps)
        psnrMin, psnrMax = min(allPsnrs), max(allPsnrs)
        marginBpp = (bppMax - bppMin) * 0.05 if bppMax > bppMin else 0.01
        marginPsnr = (psnrMax - psnrMin) * 0.05 if psnrMax > psnrMin else 0.1
        plt.xlim(bppMin - marginBpp, bppMax + marginBpp)
        plt.ylim(psnrMin - marginPsnr, psnrMax + marginPsnr)

    plt.title(f"Rate-Distortion Curves - {title}")
    plt.xlabel("Average BPP")
    plt.ylabel("Average PSNR (dB)")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fontsize='small', frameon=True)

    if bdRateTable:
        colLabels = ["Config", "BD-Rate (%)"]
        plt.table(cellText=bdRateTable, colLabels=colLabels, cellLoc='left', colLoc='center',
                  loc='lower right', bbox=[0.76, 0.04, 0.22, 0.35], edges='closed')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def printBdRateTable(rdData, reference="anchor", datasetName="Dataset"):
    output = []
    output.append(f"BD-Rate comparison to reference: {reference} - {datasetName}")
    output.append(f"{'Config':<10} | {'BD-Rate (%)':>10}")
    output.append("-" * 28)
    refData = rdData.get(reference)
    if refData is None:
        output.append("Reference config not found.")
        return "\n".join(output)
    combos = sorted(rdData.keys(), key=comboSortKey)
    for config in combos:
        if config == reference:
            continue
        bdv = bdRate(refData, rdData[config])
        if bdv is None:
            output.append(f"{config:<10} | {'N/A':>10}")
        else:
            bdRateValue, _ = bdv
            output.append(f"{config:<10} | {bdRateValue:10.4f}")
    return "\n".join(output)

# ---------------- Leitura e Processamento ----------------
files = [f for f in os.listdir('.') if os.path.isfile(f)]
grouped_by_patch = defaultdict(list)

for fname in files:
    m = filePattern.match(fname)
    if m:
        dataset, patch, frames, seqs = m.groups()
        key = f"{patch}_{frames}_{seqs}"
        grouped_by_patch[key].append((dataset, fname))

all_results = {}

for patch_key, dataset_files in grouped_by_patch.items():
    outdir = patch_key
    os.makedirs(outdir, exist_ok=True)

    rdCurves_data = {}
    patch_results = {}

    for dataset in ['HEVC-B', 'UVG']:
        data = defaultdict(lambda: defaultdict(list))
        for ds_name, filepath in dataset_files:
            if ds_name != dataset:
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
                    data[currentConfig][currentQuality].append((psnr, bpp))
        rdCurves = prepareRdData(data)
        rdCurves_data[dataset] = rdCurves

        # BD-Rate
        dataset_bd_rates = {}
        refData = rdCurves.get("anchor")
        if refData is not None:
            for config in sorted(rdCurves.keys(), key=comboSortKey):
                if config == "anchor":
                    continue
                bdResult = bdRate(refData, rdCurves[config])
                if bdResult is not None:
                    bdRateValue, _ = bdResult
                    dataset_bd_rates[config] = bdRateValue
        patch_results[dataset] = dataset_bd_rates

        # Plot dataset
        plotRdCurves(dataset, rdCurves, os.path.join(outdir, f"{dataset}.png"), reference="anchor")
        txt = printBdRateTable(rdCurves, reference="anchor", datasetName=dataset)
        with open(os.path.join(outdir, "BD-Rate.txt"), 'a') as f:
            f.write(txt + "\n\n")

    # ---------------- Overall ----------------
    hevc_rd = rdCurves_data.get('HEVC-B', {})
    uvg_rd = rdCurves_data.get('UVG', {})

    overallCurves = build_overall_curves_pointwise(hevc_rd, uvg_rd)
    hevc_bd = patch_results.get('HEVC-B', {})
    uvg_bd = patch_results.get('UVG', {})
    overall_bd_rates = compute_overall_bdrates(hevc_bd, uvg_bd)

    bd_rows_overall = [[cfg, f"{val:+.4f} %"] for cfg, val in sorted(overall_bd_rates.items(), key=lambda x: comboSortKey(x[0]))]
    plotRdCurves("Overall", overallCurves, os.path.join(outdir, "Overall.png"), reference="anchor", bdRateTable=bd_rows_overall)

    with open(os.path.join(outdir, "BD-Rate.txt"), 'a') as f:
        f.write("BD-Rate comparison to reference: anchor - Overall (precomputed average)\n")
        f.write(f"{'Config':<10} | {'BD-Rate (%)':>10}\n")
        f.write("-" * 28 + "\n")
        for cfg, val in sorted(overall_bd_rates.items(), key=lambda x: comboSortKey(x[0])):
            f.write(f"{cfg:<10} | {val:10.4f}\n")
        f.write("\n")

    patch_results['Overall'] = overall_bd_rates
    all_results[patch_key] = patch_results

# ---------------- Consolidated BD-rate dataframe ----------------
def create_bdrate_dataframe(all_results):
    w_values = [4, 8, 12, 16]
    a_values = [4, 8, 12, 16]
    data = []
    for w in w_values:
        for a in a_values:
            config_name = f"a{a}_w{w}"
            row = {
                'W': w,
                'A': a,
                'HEVC-B_t1': 0.000,
                'HEVC-B_t2': 0.000,
                'UVG_t1': 0.000,
                'UVG_t2': 0.000,
                'Average_t1': 0.000,
                'Average_t2': 0.000
            }
            hevc_b_values = []
            uvg_values = []
            for patch_key, datasets_data in all_results.items():
                if 'HEVC-B' in datasets_data and config_name in datasets_data['HEVC-B']:
                    hevc_b_values.append(datasets_data['HEVC-B'][config_name])
                if 'UVG' in datasets_data and config_name in datasets_data['UVG']:
                    uvg_values.append(datasets_data['UVG'][config_name])
            if hevc_b_values:
                hevc_avg = sum(hevc_b_values) / len(hevc_b_values)
                row['HEVC-B_t1'] = hevc_avg
                row['HEVC-B_t2'] = hevc_avg
            if uvg_values:
                uvg_avg = sum(uvg_values) / len(uvg_values)
                row['UVG_t1'] = uvg_avg
                row['UVG_t2'] = uvg_avg
            if row['HEVC-B_t1'] != 0.000 or row['UVG_t1'] != 0.000:
                row['Average_t1'] = (row['HEVC-B_t1'] + row['UVG_t1']) / 2
                row['Average_t2'] = (row['HEVC-B_t2'] + row['UVG_t2']) / 2
            data.append(row)
    return pd.DataFrame(data)

def generate_latex_table(df, table_label="tab:analysis_bw"):
    latex_code = """\\begin{tabular}{c|c||c|c|c|c|c|c}
         \\hline
         \\hline
         \\multicolumn{2}{c||}{\\textbf{BW}} & 
            \\multicolumn{2}{|c|}{\\textbf{HEVC-B}}& \\multicolumn{2}{|c}{\\textbf{UVG}} & \\multicolumn{2}{|c}{\\textbf{Average}}   \\\\
            \\hline
          \\textit{W} & \\textit{A} & $t_1$&$t_2$&$t_1$&$t_2$&$t_1$&$t_2$ \\\\
        \\hline
        \\hline
"""
    for _, row in df.iterrows():
        latex_code += f"         {int(row['W']):2d} & {int(row['A']):2d} & "
        latex_code += f"{row['HEVC-B_t1']:6.3f}& {row['HEVC-B_t2']:6.3f}& "
        latex_code += f"{row['UVG_t1']:6.3f}& {row['UVG_t2']:6.3f}& "
        latex_code += f"{row['Average_t1']:6.3f}& {row['Average_t2']:6.3f}\\\\ \\hline\n"
    latex_code += """                        
         \\hline
         \\hline
    \\end{tabular}    
    \\label{""" + table_label + """}
\\end{table}"""
    return latex_code

df_bdrate = create_bdrate_dataframe(all_results)
df_bdrate.to_csv("consolidated_bdrate_results.csv", index=False)
print(df_bdrate.to_string(index=False))

latex_table = generate_latex_table(df_bdrate)
with open("bdrate_table.tex", 'w') as f:
    f.write(latex_table)

print("\nProcessing completed. Results saved in respective directories and consolidated files created.")
