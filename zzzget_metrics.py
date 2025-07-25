import os
import re
import subprocess
import pandas as pd
from collections import defaultdict
from itertools import product

frames = 32

activationBits = [8, 16, 24, 32]
weightBits = [8, 16, 24, 32]
combos = ["anchor"] + [f"a{a}_w{w}" for a in activationBits for w in weightBits]
folders = ["HEVC-B", "UVG"]

qualities = list(range(1, 10))
saveDir = "sheets"

def writeLog(folder, combo, quality, message):
    cleanedMessage = re.sub(
        r"^Warning: no checkpoint provided; using pretrained weights only\.\n?",
        "",
        message,
        flags=re.MULTILINE,
    )
    cleanedMessage = re.sub(
        r"^\[.*?INFO.*?\].*\n?", "", cleanedMessage, flags=re.MULTILINE
    )
    cleanedMessage = re.sub(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - Quant - INFO - (Unsupported op type .+|Selecting DefaultOpInstanceConfigGenerator .+)$\n?",
        "",
        cleanedMessage,
        flags=re.MULTILINE,
    )
    cleanedMessage = re.sub(r"\n{2,}", "\n\n", cleanedMessage).strip()

    path = f"NEW_{folder}_{frames}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[OK] {folder} {combo} {quality}:\n\n")
        f.write(cleanedMessage)
        f.write("\n\n")

def hasRunAlready(folder, combo, quality):
    path = f"NEW_{folder}_{frames}"
    if not os.path.exists(path):
        return False
    header = f"[OK] {folder} {combo} {quality}:"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == header:
                return True
    return False

def run(folder, combo, quality):
    if hasRunAlready(folder, combo, quality):
        print(f"[SKIP] {folder} {combo} quality {quality} (already done)")
        return

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
        encodingPath = f"encodings/{combo}/{quality}"
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

    print(f"[RUN] {folder} {combo} quality {quality}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    writeLog(folder, combo, quality, result.stdout)

def processAll():
    for folder in folders:
        path = f"NEW_{folder}_{frames}"
        completed = set()

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.match(
                        r"\[OK\] ([\w\-]+) (anchor|a\d+_w\d+) (\d+):", line
                    )
                    if match:
                        _, combo, quality = match.groups()
                        completed.add((combo, int(quality)))
        else:
            open(path, "w").close()

        for quality in qualities:
            if ("anchor", quality) not in completed:
                run(folder, "anchor", quality)
            else:
                print(f"[SKIP] {folder} anchor quality {quality}. Done")

            for a, w in product(activationBits, weightBits):
                combo = f"a{a}_w{w}"
                if (combo, quality) in completed:
                    print(f"[SKIP] {folder} {combo} quality {quality}. Done")
                    continue
                run(folder, combo, quality)

def parseLogs():
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    headerRegex = re.compile(r"\[OK\] ([\w\-]+) (anchor|a\d+_w\d+) (\d+):")
    seqRegex = re.compile(r"Seq ([\w\d]+_\d+x\d+_\d+) .*?PSNR: (-?[\d.]+) dB, BPP: ([\d.]+)")

    for folder in folders:
        path = f"NEW_{folder}_{frames}"

        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"(?=\[OK\])", content)

        for block in blocks:
            headerMatch = headerRegex.search(block)
            if not headerMatch:
                continue

            dataset, combo, quality = headerMatch.groups()

            for seqMatch in seqRegex.finditer(block):
                seq, psnr, bpp = seqMatch.groups()
                data[dataset][combo][seq].append((int(quality), float(psnr), float(bpp)))

    return data

def save(data):
    os.makedirs(saveDir, exist_ok=True)
    xlsxPath = os.path.join(saveDir, "NEW.xlsx")

    with pd.ExcelWriter(xlsxPath, engine="xlsxwriter") as writer:
        for combo in combos:
            if all(combo not in data[ds] for ds in folders):
                continue

            allRows = []

            for dataset in folders:
                if combo not in data[dataset]:
                    continue

                for seq, qualityList in data[dataset][combo].items():
                    row = {"DATASET": dataset, "SEQUENCE": seq}
                    qualityDict = {q: (psnr, bpp) for q, psnr, bpp in qualityList}

                    for q in qualities:
                        if q in qualityDict:
                            psnr, bpp = qualityDict[q]
                            row[f"Quality {q} - PSNR"] = str(psnr)
                            row[f"Quality {q} - bpp"] = str(bpp)

                    allRows.append(row)

            if not allRows:
                continue

            df = pd.DataFrame(allRows)
            df.sort_values(by=["DATASET", "SEQUENCE"], inplace=True)

            emptyRow = {col: "" for col in df.columns}
            df = pd.concat([df, pd.DataFrame([emptyRow])], ignore_index=True)

            averages = []
            for label in ["HEVC-B", "UVG", "ALL"]:
                if label == "ALL":
                    filteredDf = df[~df["DATASET"].isin(["", "AVG", "AVG HEVC-B", "AVG UVG"])]
                    datasetLabel = "AVG"
                else:
                    filteredDf = df[df["DATASET"] == label]
                    datasetLabel = f"AVG {label}"

                avgRow = {"DATASET": datasetLabel, "SEQUENCE": ""}

                for q in qualities:
                    psnrCol = f"Quality {q} - PSNR"
                    bppCol = f"Quality {q} - bpp"

                    if psnrCol in filteredDf.columns and bppCol in filteredDf.columns:
                        psnrVals = pd.to_numeric(filteredDf[psnrCol], errors="coerce")
                        bppVals = pd.to_numeric(filteredDf[bppCol], errors="coerce")

                        avgRow[psnrCol] = f"{psnrVals.mean():.4f}"
                        avgRow[bppCol] = f"{bppVals.mean():.8f}"
                    else:
                        avgRow[psnrCol] = ""
                        avgRow[bppCol] = ""

                averages.append(avgRow)

            df = pd.concat([df, pd.DataFrame(averages)], ignore_index=True)

            dfNoDataset = df.copy()
            dfNoDataset.loc[dfNoDataset.duplicated("DATASET"), "DATASET"] = ""

            dfNoDataset.to_excel(writer, sheet_name=combo, index=False, startrow=2, header=False)

            worksheet = writer.sheets[combo]
            workbook = writer.book

            boldCenterFormat = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
            boldFormat = workbook.add_format({'bold': True, 'align': 'center'})
            centerFormat = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

            worksheet.write(0, 0, "DATASET", boldCenterFormat)
            worksheet.write(0, 1, "SEQUENCE", boldCenterFormat)

            for i, q in enumerate(qualities):
                colStart = 2 + i * 2
                worksheet.merge_range(0, colStart, 0, colStart + 1, f"Quality {q}", boldCenterFormat)
                worksheet.write(1, colStart, "PSNR", boldFormat)
                worksheet.write(1, colStart + 1, "bpp", boldFormat)

            currentDataset = None
            startRow = None
            count = 0

            for i, (_, row) in enumerate(df.iterrows()):
                dataset = row["DATASET"]
                if dataset != currentDataset:
                    if currentDataset is not None and count > 0:
                        endRow = startRow + count - 1
                        if startRow == endRow:
                            worksheet.write(startRow + 2, 0, currentDataset, boldCenterFormat)
                        else:
                            worksheet.merge_range(startRow + 2, 0, endRow + 2, 0, currentDataset, boldCenterFormat)
                    currentDataset = dataset
                    startRow = i
                    count = 1
                else:
                    count += 1

            if currentDataset is not None and count > 0:
                endRow = startRow + count - 1
                if startRow == endRow:
                    worksheet.write(startRow + 2, 0, currentDataset, boldCenterFormat)
                else:
                    worksheet.merge_range(startRow + 2, 0, endRow + 2, 0, currentDataset, boldCenterFormat)

            worksheet.set_column("A:A", 10, centerFormat)
            worksheet.set_column("B:B", 30, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
            worksheet.set_column("C:Z", 15, centerFormat)

    print(f"[✓] Generated: {xlsxPath}")

if __name__ == "__main__":
    processAll()
    parsedData = parseLogs()
    save(parsedData)
