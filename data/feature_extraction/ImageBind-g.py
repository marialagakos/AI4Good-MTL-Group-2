import os
import torch
import ffmpeg
import pandas as pd
import h5py
from tqdm import tqdm
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind.data import (
    load_and_transform_text,
    load_and_transform_vision_data,
    load_and_transform_audio_data,
)

# Setup
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"  # For ffmpeg

# Paths
TSV_PATH = "/Users/catherinamedeiros/Documents/mila/data_algonaut/friends_s01e01a.tsv" #-> need to fix for the cluster 
MKV_PATH = "/Users/catherinamedeiros/Documents/mila/data_algonaut/friends_s01e01a.mkv" #-> need to fix for the cluster 
OUTPUT_DIR = "/Users/catherinamedeiros/Documents/mila/data_algonaut" #-> need to fix for the cluster 
CHECKPOINT_PATH = ".checkpoints/imagebind_huge.pth"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "friends_s01e01a.pt")

# Config
TR_DURATION = 1.49
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = imagebind_model.imagebind_huge(pretrained=False)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.to(DEVICE)
model.eval()


def extract_middle_frame(video_path, out_path, start):
    ffmpeg.input(video_path, ss=start + TR_DURATION / 2).filter('scale', 224, 224) \
        .output(out_path, vframes=1).overwrite_output().run(quiet=True)


def extract_audio_segment(video_path, out_path, start):
    ffmpeg.input(video_path, ss=start, t=TR_DURATION).output(out_path, ac=1, ar='16000') \
        .overwrite_output().run(quiet=True)


def get_transcript(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    return df['text_per_tr'].fillna('').tolist()


def load_fmri_chunks(h5_file, device):
    chunks = []
    with h5py.File(h5_file, 'r') as f:
        for key in f:
            chunks.append(torch.tensor(f[key][()]).to(device))
    return chunks


def align_modalities_with_fmri(embeddings_dict, fmri_array, delay_sec=3.0):
    shift = int(round(delay_sec / TR_DURATION))
    T = min(
        embeddings_dict["text"].shape[0],
        embeddings_dict["vision"].shape[0],
        embeddings_dict["audio"].shape[0],
        fmri_array.shape[0] - shift
    )
    return {
        "text": embeddings_dict["text"][:T],
        "vision": embeddings_dict["vision"][:T],
        "audio": embeddings_dict["audio"][:T],
        "fmri": torch.tensor(fmri_array[shift:shift + T])
    }


def get_imagebind_embeddings(text, frame_path, audio_path):
    inputs = {
        ModalityType.TEXT: load_and_transform_text([text], device=DEVICE),
        ModalityType.VISION: load_and_transform_vision_data([frame_path], device=DEVICE),
        ModalityType.AUDIO: load_and_transform_audio_data([audio_path], device=DEVICE)
    }

    with torch.no_grad():
        embeddings = model(inputs)

    return (
        embeddings[ModalityType.TEXT],
        embeddings[ModalityType.VISION],
        embeddings[ModalityType.AUDIO]
    )


def process_file(tsv_file, mkv_file, out_file):
    transcript = get_transcript(tsv_file)
    text_embeddings, vision_embeddings, audio_embeddings = [], [], []

    for i, text in tqdm(enumerate(transcript), total=len(transcript), desc=os.path.basename(tsv_file)):
        start_time = i * TR_DURATION
        frame_file = f"tmp_frame_{i}.jpg"
        audio_file = f"tmp_audio_{i}.wav"

        extract_middle_frame(mkv_file, frame_file, start_time)
        extract_audio_segment(mkv_file, audio_file, start_time)

        try:
            t, v, a = get_imagebind_embeddings(text, frame_file, audio_file)
            text_embeddings.append(t.cpu())
            vision_embeddings.append(v.cpu())
            audio_embeddings.append(a.cpu())
        except Exception as e:
            print(f"Skipping TR {i} due to error: {e}")
        finally:
            os.remove(frame_file)
            os.remove(audio_file)

    embeddings_dict = {
        "text": torch.cat(text_embeddings, dim=0),
        "vision": torch.cat(vision_embeddings, dim=0),
        "audio": torch.cat(audio_embeddings, dim=0),
    }

    # Infer fMRI segment from filename
    seg_idx = int(os.path.basename(tsv_file).split("_")[-1][1:].split(".")[0])
    fmri_chunks = load_fmri_chunks(os.path.join(OUTPUT_DIR, "friends_s01e01a.h5"), DEVICE)
    fmri_segment = fmri_chunks[seg_idx]

    aligned = align_modalities_with_fmri(embeddings_dict, fmri_segment)

basename = os.path.splitext(os.path.basename(tsv_file))[0]  # e.g., 'friends_s01e01a'

for modality in ["text", "vision", "audio"]:
    data = aligned[modality]
    out_path = os.path.join(OUTPUT_DIR, f"{basename}_{modality}_Image_Bind.h5")

    if os.path.exists(out_path):
        print(f"[SKIP] {out_path} already exists.")
        continue

    with h5py.File(out_path, "w") as f:
        f.create_dataset("embedding", data=data.cpu().numpy())
        f.attrs["modality"] = modality
        f.attrs["episode"] = basename
        f.attrs["shape"] = data.shape
        print(f"[WRITE] Saved {out_path}")

# Run processing
process_file(TSV_PATH, MKV_PATH, OUTPUT_PATH)
