#!/usr/bin/env python3
"""
Descarga un video de YouTube y recorta un tramo (por defecto 4:45 -> 5:50).

Elige AUTOMÁTICAMENTE la resolución nativa: ignora los formatos que YouTube marca
como "AI-upscaled" (agrandados con IA), porque tienen detalle inventado y no sirven
para entrenar un modelo de super-resolución.

Uso:
    python descargar_clip.py
    python descargar_clip.py --url https://youtu.be/ZrSrRGKu0_Y --start 4:45 --end 5:50 --out clip.mp4
Requiere: yt-dlp (pip install yt-dlp) y ffmpeg en el PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def to_seconds(t: str) -> float:
    parts = [float(p) for p in t.split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def ytdlp(*args) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--no-warnings", *args]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://youtu.be/ZrSrRGKu0_Y")
    ap.add_argument("--start", default="4:45")
    ap.add_argument("--end", default="5:50")
    ap.add_argument("--out", default="clip_especial_humor_congreso_240p.mp4")
    ap.add_argument("--max-height", type=int, default=0,
                    help="0 = la mayor resolución NO escalada con IA; o fija un tope (p. ej. 360)")
    a = ap.parse_args()

    # En Mac con Python de python.org, yt-dlp necesita los certificados de certifi
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    # 1) Elegir formato nativo (sin "AI-upscaled")
    info = json.loads(subprocess.check_output(ytdlp("-J", "--skip-download", a.url)))
    print(f"Video: {info['title']} | duración {info.get('duration_string')} | {info.get('upload_date')}")
    cands = [f for f in info["formats"]
             if f.get("vcodec", "none") != "none" and f.get("height")
             and f.get("ext") == "mp4" and str(f.get("vcodec", "")).startswith("avc1")
             and "upscaled" not in (f.get("format_note") or "").lower()
             and (not a.max_height or f["height"] <= a.max_height)]
    if not cands:
        sys.exit("No se encontró un formato H.264 nativo. Prueba con --max-height o revisa el enlace.")
    best = max(cands, key=lambda f: (f["height"], f.get("tbr") or 0))
    print(f"Formato elegido: {best['format_id']} -> {best['width']}x{best['height']} ({best.get('format_note')})")
    ignored = sorted({f"{f['height']}p" for f in info["formats"]
                      if "upscaled" in (f.get("format_note") or "").lower() and f.get("height")},
                     key=lambda s: int(s[:-1]))
    print(f"Ignorados por ser 'AI-upscaled': {', '.join(ignored) or 'ninguno'}")

    # 2) Descargar a una carpeta temporal (fuera del proyecto)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        subprocess.run(ytdlp("-f", f"{best['format_id']}+bestaudio[ext=m4a]/{best['format_id']}+bestaudio",
                             "--merge-output-format", "mp4", "-o", str(raw), a.url), check=True)

        # 3) Recorte exacto (re-codifica con calidad casi sin pérdida: crf 12)
        t0, t1 = to_seconds(a.start), to_seconds(a.end)
        out = Path(a.out)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(t0), "-i", str(raw), "-t", str(t1 - t0),
                        "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "160k", str(out)], check=True)

    p = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
         "-of", "json", str(out)]))
    v = next(s for s in p["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in p["streams"])
    print(f"\n[OK] {out.resolve()}")
    print(f"     {v['width']}x{v['height']} @ {v['r_frame_rate']} | {float(p['format']['duration']):.1f} s | "
          f"audio: {'sí' if has_audio else 'NO'}")


if __name__ == "__main__":
    main()
