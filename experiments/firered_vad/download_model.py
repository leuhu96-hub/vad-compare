"""Tải 4 file ONNX + cmvn.ark của FireRedVAD vào ./models (không cần package fireredvad).

    python download_model.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/pretrained_models/onnx_models/"
FILES = ["fireredvad_vad.onnx", "fireredvad_stream_vad.onnx",
         "fireredvad_stream_vad_with_cache.onnx", "cmvn.ark"]


def main():
    dst_dir = os.path.join(HERE, "models")
    os.makedirs(dst_dir, exist_ok=True)
    for f in FILES:
        dst = os.path.join(dst_dir, f)
        if os.path.exists(dst):
            print(f"[skip] {f}")
            continue
        print(f"[get ] {f}")
        try:
            urllib.request.urlretrieve(BASE + f, dst + ".part")
            os.replace(dst + ".part", dst)
        except Exception as e:
            print(f"       LỖI: {e}\n       -> tải tay {BASE + f} vào {dst}")


if __name__ == "__main__":
    main()
