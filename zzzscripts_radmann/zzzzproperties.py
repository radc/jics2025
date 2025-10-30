frames           = 96
sequences        = 1000
bits             = [4, 8, 12, 16]

patchSize        = ["256", "256"]
folders          = ["HEVC-B", "UVG"]

logFiles         = {
                    "HEVC-B": f"HEVC-B_{patchSize[0]}_{frames}_{sequences}",
                    "UVG": f"UVG_{patchSize[0]}_{frames}_{sequences}"
                   }

encodingsPath    = f"encodings_{patchSize[0]}x{patchSize[1]}_{sequences}"