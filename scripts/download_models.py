"""Download open-source VAD model files into ./models.

    python scripts/download_models.py            # silero + firered
    python scripts/download_models.py --only silero

TEN VAD and WebRTC come as pip packages (see README); pyannote downloads from Hugging Face
on first use. Your own YAMNet model: copy it to models/yamnet_vad.tflite (or .onnx).
"""
from __future__ import annotations

import argparse
import os
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
FILES = {
    "silero": [
        ("https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
         "silero_vad.onnx"),
    ],
    "firered": [
        ("https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/fireredvad_vad.onnx",
         "firered/fireredvad_vad.onnx"),
        ("https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/fireredvad_stream_vad.onnx",
         "firered/fireredvad_stream_vad.onnx"),
        ("https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/cmvn.ark",
         "firered/cmvn.ark"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FILES), nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for name in a.only or FILES:
        for url, rel in FILES[name]:
            dst = os.path.normpath(os.path.join(ROOT, rel))
            if os.path.exists(dst) and not a.force:
                print(f"[skip] {rel} (đã có)")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"[get ] {rel}  <-  {url}")
            try:
                urllib.request.urlretrieve(url, dst + ".part")
                os.replace(dst + ".part", dst)
            except Exception as e:
                print(f"       FAILED: {e}\n       -> tải tay file trên và đặt vào {dst}")


if __name__ == "__main__":
    main()
