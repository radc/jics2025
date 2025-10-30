import os
import re
import pandas as pd
from collections import defaultdict
from itertools import product
import xlsxwriter

frames = 32

as_ = [8, 16, 24, 32]
ws_ = [8, 16, 24, 32]
combos = [f"a{a}_w{w}" for a, w in product(as_, ws_)]
combos.insert(0, "anchor")

QUALIDADES = list(range(1, 10))
DATASETS = [f"NEW_HEVC-B_{frames}", f"NEW_UVG_{frames}"]

entry_re = re.compile(r"Seq ([\w\d]+_\d+x\d+_\d+) .*?PSNR: (-?[\d.]+) dB, BPP: (-?[\d.]+)")
todos_dados = {combo: defaultdict(lambda: defaultdict(dict)) for combo in combos}

for dataset in DATASETS:
    LOG_PATH = dataset
    datasetString = dataset.removeprefix("NEW_").removesuffix(f"_{frames}")

    header_re = re.compile(rf"\[OK\] {datasetString} (anchor|a\d+_w\d+) (\d+):")

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        linhas = f.readlines()

    combo_atual = None
    qualidade_atual = None

    for linha in linhas:
        header = header_re.match(linha)

        if header:
            combo_atual = header.group(1)
            qualidade_atual = int(header.group(2))
            continue

        if combo_atual and qualidade_atual:
            match = entry_re.search(linha)
            if match:
                seq = match.group(1)
                psnr = float(match.group(2))
                bpp = float(match.group(3))
                todos_dados[combo_atual][datasetString][seq][qualidade_atual] = (psnr, bpp)

SAVE_DIR = "sheets"
os.makedirs(SAVE_DIR, exist_ok=True)
xlsx_path = os.path.join(SAVE_DIR, "NEW.xlsx")

with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
    for combo, dados_combo in todos_dados.items():
        if not dados_combo:
            continue

        linhas_gerais = []

        for dataset, seqs in dados_combo.items():
            for seq, qualidades in seqs.items():
                linha = {"DATASET": dataset, "SEQUENCE": seq}
                for q in QUALIDADES:
                    if q in qualidades:
                        psnr, bpp = qualidades[q]
                        linha[f"Quality {q} - PSNR"] = f"{psnr:.4f}"
                        linha[f"Quality {q} - bpp"] = f"{bpp:.8f}"
                linhas_gerais.append(linha)

        if not linhas_gerais:
            continue

        df = pd.DataFrame(linhas_gerais)
        df.sort_values(by=["DATASET", "SEQUENCE"], inplace=True)

        empty_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)

        medias = []

        for label in ["HEVC-B", "UVG", "ALL"]:
            if label == "ALL":
                df_filtrado = df[~df["DATASET"].isin(["", "AVG", "AVG HEVC-B", "AVG UVG"])]
                dataset_label = "AVG"

            else:
                df_filtrado = df[df["DATASET"] == label]
                dataset_label = f"AVG {label}"

            avg_row = {"DATASET": dataset_label, "SEQUENCE": ""}

            for q in QUALIDADES:
                psnr_col = f"Quality {q} - PSNR"
                bpp_col = f"Quality {q} - bpp"

                psnr_vals = pd.to_numeric(df_filtrado[psnr_col], errors='coerce')
                bpp_vals = pd.to_numeric(df_filtrado[bpp_col], errors='coerce')

                avg_row[psnr_col] = f"{psnr_vals.mean():.4f}"
                avg_row[bpp_col] = f"{bpp_vals.mean():.8f}"
            
            medias.append(avg_row)

        df = pd.concat([df, pd.DataFrame(medias)], ignore_index=True)

        df_no_dataset = df.copy()
        df_no_dataset.loc[df_no_dataset.duplicated("DATASET"), "DATASET"] = ""

        df_no_dataset.to_excel(writer, sheet_name=combo, index=False, startrow=2, header=False)

        worksheet = writer.sheets[combo]
        workbook = writer.book

        bold_center = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
        bold = workbook.add_format({'bold': True, 'align': 'center'})
        center = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        worksheet.write(0, 0, "DATASET", bold_center)
        worksheet.write(0, 1, "SEQUENCE", bold_center)

        for i, q in enumerate(QUALIDADES):
            col_start = 2 + i * 2
            worksheet.merge_range(0, col_start, 0, col_start + 1, f"Quality {q}", bold_center)
            worksheet.write(1, col_start, "PSNR", bold)
            worksheet.write(1, col_start + 1, "bpp", bold)

        current_dataset = None
        start_row = None
        count = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            dataset = row["DATASET"]
            if dataset != current_dataset:

                if current_dataset is not None and count > 0:
                    end_row = start_row + count - 1

                    if start_row == end_row:
                        worksheet.write(start_row + 2, 0, current_dataset, bold_center)
                        
                    else:
                        worksheet.merge_range(start_row + 2, 0, end_row + 2, 0, current_dataset, bold_center)

                current_dataset = dataset
                start_row = i
                count = 1
            else:
                count += 1

        if current_dataset is not None and count > 0:
            end_row = start_row + count - 1
            if start_row == end_row:
                worksheet.write(start_row + 2, 0, current_dataset, bold_center)
            else:
                worksheet.merge_range(start_row + 2, 0, end_row + 2, 0, current_dataset, bold_center)

        worksheet.set_column("A:A", 10, center)
        worksheet.set_column("B:B", 30, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
        worksheet.set_column("C:Z", 15, center)

print(f"[✓] Gerado: {xlsx_path}")
