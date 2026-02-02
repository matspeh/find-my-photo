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

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"


@st.cache_resource
def load_clip():
    """Load CLIP model and processor once and cache."""
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

    # Load model and run search
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with st.spinner("Loading CLIP model..."):
        model, processor = load_clip()
        model = model.to(device)

    with st.spinner("Searching images..."):
        top_paths, top_scores = search_images(
            search_query, image_paths, model, processor, device, top_k=5
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
