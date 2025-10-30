import subprocess
import re

OUTPUT_RESULTS_FILE = "ZRESULTS.txt"

def filter_stdout(text):
    # Padrão para qualquer resolução e framerate
    padrao = re.compile(
        r'^\[Epoch \d+\] Seq [A-Za-z0-9_]+_\d+x\d+_\d+ \(#\d+\) → Loss: .*?, MSE: .*?, PSNR: .*?, BPP: .*?$'
    )
    return [line for line in text.splitlines() if padrao.match(line)]

def save_formatted_line(a, w, calib_res, calib_seq, test_frames, dataset, seq, quality, bpp, psnr):
    with open(OUTPUT_RESULTS_FILE, "a") as f:
        print(f"{a}, {w}, {calib_res}, {calib_seq}, {test_frames}, {dataset}, {quality}, {seq}, {bpp}, {psnr}", file=f)
        f.flush()

def extract_seq_psnr_bpp(line):
    pattern = re.compile(
        r'Seq\s+(?P<seq>[A-Za-z0-9_]+)_\d+x\d+_\d+.*?PSNR:\s+(?P<psnr>[\d\.]+)\s+dB,\s+BPP:\s+(?P<bpp>[\d\.]+)'
    )
    match = pattern.search(line)
    if match:
        return {
            "seq": match.group("seq"),
            "psnr": float(match.group("psnr")),
            "bpp": float(match.group("bpp"))
        }
    return None


def process_results(a, w, calib_res, calib_seq, frames, dataset, quality, results):
    for r in results :
        data_dict = extract_seq_psnr_bpp(r)
        save_formatted_line(a, w, calib_res, calib_seq, frames, dataset, data_dict["seq"], quality, data_dict["bpp"], data_dict["psnr"])

def already_processed(a, w, calib_res, calib_seq, frames, dataset, quality):
    try:
        with open(OUTPUT_RESULTS_FILE, "r") as f:
            for line in f:
                if f"{a}, {w}, {calib_res}, {calib_seq}, {frames}, {dataset}, {quality}" in line:
                    return True
    except FileNotFoundError:
        return False
    return False

FRAMES = 32
VIDEOS = [
    {
        "HEVC-B" : [
            "BasketballDrive_1920x1080_50",
            "BQTerrace_1920x1080_60",
            "Cactus_1920x1080_50",
            "Kimono1_1920x1080_24",
            "ParkScene_1920x1080_24"
        ],
        "UVG" : [
            "Beauty_1920x1080_120",
            "Bosphorus_1920x1080_120",
            "HoneyBee_1920x1080_120",
            "Jockey_1920x1080_120",
            "ReadySteadyGo_1920x1080_120",
            "ShakeNDry_1920x1080_120",
            "YachtRide_1920x1080_120"
        ]
    }
]

DATASETS = ["UVG", "HEVC-B"]


for calib_res in [256, 128] :
    for calib_seq in [1000, 200] :
        for frames in ["32","96"] :
            # for w in range(4,17,4) :
            for w in [8] :
                # for a in range(4,17,4) :
                for a in [8] :
                    for quality in range(1,10) :
                        for dataset in DATASETS :
                            if already_processed(a, w, calib_res, calib_seq, frames, dataset, quality) :
                                print(f"SKIPPING {a=}, {w=}, {calib_res=}, {calib_seq}, {frames}, {dataset}, {quality}")
                                continue
                            
                            cmd = [
                                "python", "ztest_video.py",
                                "-m", "ssf2020",
                                "-d", f"/data/Radmann/{dataset}",
                                "--patch-size", "1920", "1024",
                                # "--calc-params",
                                "--cuda",
                                "--num-frames", frames,
                                "--quality", str(quality),
                                "--test-batch-size", "4" if frames == "32" else "2",
                                "--aimet-load-encodings",
                                "--aimet-path-encodings", f"/home/ruhan/jics2025/encodings_{calib_res}x{calib_res}_{calib_seq}/a{a}_w{w}/{quality}",
                                "--ignore-sequence-folder",
                            ]

                            print(f"NOW RUNNING {a=}, {w=}, {calib_res=}, {calib_seq}, {frames}, {dataset}, {quality}", end='\r')
                            result = subprocess.run(cmd, capture_output=True, text=True)                    
                            results = filter_stdout(result.stdout)     
                            
                            if results == [] :
                                print(result.stderr)

                            process_results(a, w, calib_res, calib_seq, frames, dataset, quality, results)                    
                            print(f"DONE! {a=}, {w=}, {calib_res=}, {calib_seq=}, {frames=}, {dataset=}, {quality=}          ")



import requests

# Use o novo token que você gerar no BotFather
TOKEN = "8176289669:AAFpf0QPNU3LNWThSdX6vGMS4UdKcSIrVfw"
CHAT_ID = 1463494848
MESSAGE = "O SCRIPT TERMINOU DE EXECUTAR NESTE MOMENTO"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

payload = {
    "chat_id": CHAT_ID,
    "text": MESSAGE
}

response = requests.post(url, data=payload)

if response.status_code == 200:
    print("Mensagem enviada com sucesso!")
else:
    print(f"Erro {response.status_code}: {response.text}")