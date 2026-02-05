"""
FindMyPhoto - CLIP-powered image search in a folder.
"""
import os
from pathlib import Path

import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import tkinter as tk
from tkinter import filedialog

# Setup for easy chromadb usage later in the program
import chromadb
from chromadb.config import Settings

# Location of the chroma_db database
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "image_embeddings"


@st.cache_resource
def get_chroma_collection():
    """Create or load a persistent Chroma collection for image embeddings."""
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(allow_reset=False),
    )
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return collection

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@st.cache_resource
def load_clip():
    """Load CLIP model and processor once and cache."""
    print("load_clip")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def get_image_paths(folder_path: str) -> list[Path]:
    """Return list of image file paths in folder and all subfolders (recursive)."""
    print("get_image_paths")  # Print the name of the function
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    paths = []
    visited_dirs = set()
    for p in folder.rglob("*"):
        parent_dir = p.parent
        # Report entering this folder (once per folder)
        if parent_dir not in visited_dirs:
            print(f"Entering folder: {parent_dir}")
            visited_dirs.add(parent_dir)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)

def load_images(paths: list[Path], max_size: tuple[int, int] = (224, 224)) -> list[Image.Image]:
    """Load and optionally resize images. Returns list of PIL Images."""
    print("load_images")
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail(max_size)
            images.append(img)
        except Exception:
            continue
    return images


def search_images(query: str, image_paths: list[Path], model, processor, device, top_k: int = 5):
    """
    Encode query and images with CLIP, compute similarity, return top_k paths and scores.
    """
    if not query.strip() or not image_paths:
        return [], []

    images = load_images(image_paths)
    if not images:
        return [], []

    inputs = processor(
        text=[query],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

    # Normalize and compute similarity (cosine)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    similarity = (text_embeds @ image_embeds.T).squeeze(0)

    scores, indices = torch.topk(similarity, min(top_k, len(image_paths)))
    top_paths = [image_paths[i] for i in indices.cpu().tolist()]
    top_scores = scores.cpu().tolist()

    return top_paths, top_scores


def search_images_with_chroma(
    query: str,
    collection,
    model,
    processor,
    device,
    top_k: int = 5,
):
    print("search_images_with_chroma")
    if not query.strip():
        return [], []

    # 1. Embed the text query
    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = model.get_text_features(**text_inputs)
    text_embeds = out.pooler_output if hasattr(out, "pooler_output") else out
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # [1, D]

    # 2. Load all image embeddings + metadata from Chroma
    results = collection.get(include=["embeddings", "metadatas"])
    if not results["ids"]:
        return [], []
    
    image_embeds = torch.tensor(results["embeddings"], dtype=torch.float32, device=device)  # [N, D]
    paths = [Path(m["path"]) for m in results["metadatas"]]

    # 3. Cosine similarity via dot product (already normalized)
    similarity = (text_embeds @ image_embeds.T).squeeze(0)  # [N]

    scores, indices = torch.topk(similarity, min(top_k, image_embeds.shape[0]))
    top_paths = [paths[i] for i in indices.cpu().tolist()]
    top_scores = scores.cpu().tolist()

    return top_paths, top_scores

def index_images_in_chroma(
    image_paths: list[Path],
    model,
    processor,
    device,
    collection,
    batch_size: int = 64,
):
    print("index_images_in_chroma")
    # 1. Build the list of IDs for all images
    all_ids = [str(p.resolve()) for p in image_paths]

    # 2. Ask Chroma what it already has (in chunks to be safe)
    existing_ids = set()
    chunk_size = 1000
    for i in range(0, len(all_ids), chunk_size):
        chunk = all_ids[i:i + chunk_size]
        results = collection.get(ids=chunk, include=[])
        existing_ids.update(results["ids"])

    # 3. Filter out images that are already indexed
    new_paths = [p for p in image_paths if str(p.resolve()) not in existing_ids]
    if not new_paths:
        return  # nothing new to index

    # 4. Compute embeddings for new images (in batches)
    for i in range(0, len(new_paths), batch_size):
        batch_paths = new_paths[i:i + batch_size]
        images = load_images(batch_paths)  # you already have this helper
        if not images:
            continue

        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            out = model.get_image_features(**inputs)
        image_embeds = out.pooler_output if hasattr(out, "pooler_output") else out
        # Normalize for cosine similarity
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        ids = [str(p.resolve()) for p in batch_paths]
        metadatas = [{"path": str(p.resolve())} for p in batch_paths]

        # Convert to plain Python lists for Chroma
        collection.add(
            ids=ids,
            embeddings=image_embeds.cpu().tolist(),
            metadatas=metadatas,
        )

# Invoked each run by streamlit..
def main():
    print("main")  # Print the name of the function
    st.set_page_config(page_title="FindMyPhoto", page_icon="🔍", layout="wide")
    st.title("🔍 FindMyPhoto")
    st.caption("Search images in a folder using natural language (CLIP)")

    # Ensure session state key exists for folder path
    if "folder_path" not in st.session_state:
        st.session_state["folder_path"] = ""

    # Sidebar: folder path
    with st.sidebar:
        st.header("Settings")
        if st.button("Select Folder"):
            # Use a native folder selection dialog via tkinter
            root = tk.Tk()
            root.withdraw()  # Hide the main tkinter window
            root.attributes("-topmost", True)  # Bring dialog to front
            selected_folder = filedialog.askdirectory()
            root.destroy()

            if selected_folder:
                st.session_state["folder_path"] = selected_folder

        folder_path = st.session_state.get("folder_path", "")
        if folder_path:
            st.text(f"Selected folder:\n{folder_path}")

    # Main: search query
    search_query = st.text_input(
        "Search query",
        placeholder="e.g. a dog playing in the snow, sunset at the beach",
        label_visibility="collapsed",
    )

    if not folder_path:
        st.info("👈 Click 'Select Folder' in the sidebar to choose an image folder.")
        return

    if not Path(folder_path).is_dir():
        st.error(f"Folder not found: {folder_path}")
        return

    image_paths = get_image_paths(folder_path)
    if not image_paths:
        st.warning(f"No image files found in {folder_path}. Supported: {', '.join(IMAGE_EXTENSIONS)}")
        return

    st.sidebar.caption(f"Found {len(image_paths)} images in folder.")

    if not search_query.strip():
        st.info("Enter a search query above to find matching images.")
        return

    # Load model and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading CLIP model..."):
        model, processor = load_clip()
        model = model.to(device)

    # Ensure all images are indexed in the persistent Chroma DB
    collection = get_chroma_collection()
    with st.spinner("Indexing images (only new ones)..."):
        index_images_in_chroma(image_paths, model, processor, device, collection)

    # Use persisted embeddings for searching
    with st.spinner("Searching images..."):
        top_paths, top_scores = search_images_with_chroma(
            search_query, collection, model, processor, device, top_k=5
        )

    if not top_paths:
        st.warning("No images could be loaded or matched.")
        return

    st.subheader("Top 5 matching images")
    cols = st.columns(5)
    for i, (path, score) in enumerate(zip(top_paths, top_scores)):
        with cols[i]:
            try:
                img = Image.open(path).convert("RGB")
                st.image(img, use_container_width=True, caption=f"Score: {score:.3f}")
                st.caption(path.name)
            except Exception as e:
                st.error(f"Could not display {path.name}: {e}")


if __name__ == "__main__":
    main()
