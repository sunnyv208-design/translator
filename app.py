import io
import os
import tempfile
import time
import math

import streamlit as st
import whisper
import torch
import pandas as pd
from pydub import AudioSegment  # for chunking
from pydub.utils import which

# Manually set ffmpeg path
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"

st.set_page_config(page_title="Whisper MP3 Transcriber", layout="centered")
st.title("🎧 Whisper MP3 Transcriber (with Translation Option)")

# ------------------ Sidebar Settings ------------------
st.sidebar.header("Settings")

model_name = st.sidebar.selectbox(
    "Whisper model",
    ["small", "medium", "large"],
    index=1
)

task = st.sidebar.selectbox(
    "Task",
    ["transcribe", "translate"],  # translate = Hindi → English
    index=0
)

language = st.sidebar.text_input(
    "Input language code (e.g. hi for Hindi, en for English)",
    value=""
)

temperature = st.sidebar.slider(
    "Decoding temperature",
    0.0, 1.0, 0.0, 0.1
)

has_cuda = torch.cuda.is_available()
device = "cuda" if has_cuda else "cpu"
fp16 = True if device == "cuda" else False  # ✅ disable fp16 on CPU

st.sidebar.write(f"**Device:** `{device}`  |  **FP16:** `{fp16}`")

# ------------------ File Upload ------------------
uploaded = st.file_uploader(
    "Upload audio (mp3/wav/m4a/mp4/webm/wma/ogg/flac)",
    type=["mp3","wav","m4a","mp4","webm","wma","ogg","flac"]
)

def segments_to_df(segments):
    rows = []
    for seg in segments:
        rows.append({
            "start_sec": seg.get("start", 0.0),
            "end_sec": seg.get("end", 0.0),
            "text": seg.get("text", "").strip()
        })
    return pd.DataFrame(rows)

# ------------------ Chunked Transcription ------------------
def transcribe_in_chunks(model, audio_file_like, options, chunk_ms=30_000):
    audio = AudioSegment.from_file(audio_file_like)
    total_ms = len(audio)
    chunks = [(start, min(start + chunk_ms, total_ms)) for start in range(0, total_ms, chunk_ms)]

    full_text_parts = []
    all_segments = []

    progress = st.progress(0)
    status = st.empty()
    t0 = time.time()

    for i, (start, end) in enumerate(chunks, start=1):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_chunk:
            audio[start:end].export(tmp_chunk.name, format="mp3")
            chunk_path = tmp_chunk.name

        # Run whisper
        result = model.transcribe(chunk_path, **options)

        # Offset timestamps by chunk start
        offset_s = start / 1000.0
        for seg in result.get("segments", []):
            seg["start"] = seg.get("start", 0.0) + offset_s
            seg["end"] = seg.get("end", 0.0) + offset_s
            all_segments.append(seg)

        full_text_parts.append(result.get("text", ""))

        try: os.remove(chunk_path)
        except: pass

        # Progress bar + ETA
        frac = end / total_ms
        progress.progress(min(1.0, frac))
        elapsed = time.time() - t0
        processed_sec = end / 1000.0
        speed = processed_sec / elapsed if elapsed > 0 else 0
        remaining_sec = (total_ms/1000.0 - processed_sec) / speed if speed > 0 else 0
        eta = time.strftime("%M:%S", time.gmtime(max(0, int(remaining_sec))))
        status.write(f"Chunk {i}/{len(chunks)} — processed {processed_sec:.1f}s of {total_ms/1000:.1f}s | ETA ~ {eta}")

    progress.progress(1.0)
    status.write("✅ Transcription complete.")
    return "".join(full_text_parts).strip(), all_segments

# ------------------ Main Logic ------------------
if uploaded:
    st.audio(uploaded, format="audio/mp3")
    if st.button("📝 Start Transcription / Translation"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_in:
            tmp_in.write(uploaded.getbuffer())
            tmp_in_path = tmp_in.name

        try:
            with st.spinner(f"Loading Whisper model '{model_name}'…"):
                model = whisper.load_model(model_name, device=device)

            options = {"temperature": temperature, "fp16": fp16, "task": task}
            if language.strip():
                options["language"] = language.strip()

            start_time = time.time()
            full_text, segments = transcribe_in_chunks(model, tmp_in_path, options, chunk_ms=30_000)
            elapsed = time.time() - start_time
            st.success(f"Done in {elapsed:.1f}s ✅")

            st.subheader("Transcript / Translation")
            st.text_area("Output", value=full_text, height=350)

            if segments:
                st.subheader("Segments (timestamps)")
                df = segments_to_df(segments)
                st.dataframe(df, use_container_width=True)

                # Build SRT
                def to_srt(segments):
                    def fmt_time(sec):
                        h = int(sec // 3600)
                        m = int((sec % 3600) // 60)
                        s = int(sec % 60)
                        ms = int((sec - int(sec)) * 1000)
                        return f"{h:02}:{m:02}:{s:02},{ms:03}"
                    lines = []
                    for i, seg in enumerate(segments, start=1):
                        lines.append(str(i))
                        lines.append(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}")
                        lines.append(seg['text'].strip())
                        lines.append("")
                    return "\n".join(lines)

                srt_text = to_srt(segments)

                st.download_button(
                    "⬇️ Download transcript (.txt)",
                    data=full_text.encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded.name)[0]}_transcript.txt",
                    mime="text/plain",
                )
                st.download_button(
                    "⬇️ Download subtitles (.srt)",
                    data=srt_text.encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded.name)[0]}.srt",
                    mime="text/plain",
                )
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            try: os.remove(tmp_in_path)
            except: pass
else:
    st.info("Upload an audio file to begin.")