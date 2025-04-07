# -*- coding: utf-8 -*-
import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI # Ensure OpenAI library is installed: pip install openai
import re
import pathlib # For path manipulation
# import shutil # Not strictly needed as zipfile handles folder zipping
import time
import requests # Needed for downloading generated image if URL is returned (though using b64)
import random # To select random images for style analysis

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library" # Main directory for storing category folders
STYLE_ANALYSIS_SAMPLE_SIZE = 3 # Number of images to analyze for category style

# --- Updated Standard Category Definitions (Now with descriptions) ---
STANDARD_CATEGORIES_WITH_DESC = [
    {'name': "Text-Dominant", 'description': "Large, bold typography is the primary focus."},
    {'name': "Minimalist / Clean", 'description': "Uncluttered, simple background, few elements."},
    {'name': "Face-Focused", 'description': "Close-up, expressive human face is central."},
    {'name': "Before & After", 'description': "Divided layout showing two distinct states."},
    {'name': "Comparison / Versus", 'description': "Layout structured comparing items/ideas."},
    {'name': "Collage / Multi-Image", 'description': "Composed of multiple distinct images arranged together."},
    {'name': "Image-Focused", 'description': "A single, high-quality photo/illustration is dominant."},
    {'name': "Branded", 'description': "Prominent, consistent channel branding is the key feature."},
    {'name': "Curiosity Gap / Intrigue", 'description': "Deliberately obscures info (blurring, arrows, etc.)."},
    {'name': "High Contrast", 'description': "Stark differences in color values (e.g., brights on black)."},
    {'name': "Gradient Background", 'description': "Prominent color gradient as background/overlay."},
    {'name': "Bordered / Framed", 'description': "Distinct border around the thumbnail or key elements."},
    {'name': "Inset / PiP", 'description': "Smaller image inset within a larger one (e.g., reaction, tutorial)."},
    {'name': "Arrow/Circle Emphasis", 'description': "Prominent graphical arrows/circles drawing attention."},
    {'name': "Icon-Driven", 'description': "Relies mainly on icons or simple vector graphics."},
    {'name': "Retro / Vintage", 'description': "Evokes a specific past era stylistically."},
    {'name': "Hand-Drawn / Sketch", 'description': "Uses elements styled to look drawn or sketched."},
    {'name': "Textured Background", 'description': "Background is a distinct visual texture (paper, wood, etc.)."},
    {'name': "Extreme Close-Up (Object)", 'description': "Intense focus on a non-face object/detail."},
    {'name': "Other / Unclear", 'description': "Doesn't fit well or mixes styles heavily."}
]

# Extract just the names for convenience in some functions
STANDARD_CATEGORIES = [cat['name'] for cat in STANDARD_CATEGORIES_WITH_DESC]

# Set page configuration
st.set_page_config(
    page_title="Thumbnail Toolkit (Analyze & Generate)",
    page_icon="🖼️",
    layout="wide"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Ensure equal height for containers in a row and style them */
    .stVerticalBlock > .stHorizontalBlock > div[data-testid="column"] {
         /* Apply styling cautiously - this might affect all columns */
         /* Consider adding specific classes if more control is needed */
    }
    .thumbnail-analysis-container { /* New class for upload/analyze grid items */
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; /* Make containers equal height in a row */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Push button to bottom if needed */
    }
     .db-thumbnail-container { /* Class for library explorer items */
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        position: relative; /* Needed for absolute positioning of delete button */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; /* Make containers equal height in a row */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Push button to bottom */
    }
    .analysis-box { /* For results within a container */
        margin-top: 10px;
        padding: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        border: 1px solid #ced4da;
    }
    .analysis-box p, .analysis-box div[data-testid="stMarkdown"] p { /* Style text inside analysis box */
       margin-bottom: 5px !important; /* Force margin */
       font-size: 0.9em !important;
       line-height: 1.3 !important;
    }
    .element-checkbox label { /* Style checkbox labels */
        font-size: 0.85em;
    }
    .delete-button-container {
        margin-top: 10px;
        text-align: center;
    }
    .delete-button-container button {
       width: 80%;
       background-color: #dc3545; /* Red */
       color: white;
       border: none;
       padding: 5px 10px;
       border-radius: 4px;
       cursor: pointer;
       transition: background-color 0.2s ease;
    }
    .delete-button-container button:hover {
        background-color: #c82333; /* Darker red */
    }
    .stButton>button { /* General buttons */
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stDownloadButton>button { /* Download buttons */
        width: 100%;
    }
    /* Generated image container */
    .generated-image-container img {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .generated-image-container {
        margin-top: 15px;
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Filesystem Library Functions ----------

def sanitize_foldername(name):
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name)
    name = re.sub(r'_+', '_', name)
    if name.upper() in ["CON", "PRN", "AUX", "NUL"] or re.match(r"^(COM|LPT)[1-9]$", name.upper()):
        name = f"_{name}_"
    return name if name else "uncategorized"

def ensure_library_dir():
    pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)

def create_predefined_category_folders(category_list):
    ensure_library_dir()
    created_count = 0
    for category_name in category_list:
        sanitized_name = sanitize_foldername(category_name)
        if not sanitized_name or sanitized_name in ["uncategorized", "other_unclear"]:
            continue
        folder_path = pathlib.Path(LIBRARY_DIR) / sanitized_name
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
            except Exception as e:
                st.sidebar.warning(f"Could not create folder for '{category_name}': {e}")
    if created_count > 0:
        st.sidebar.caption(f"Created {created_count} new category folders.")

def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    ensure_library_dir()
    if not label or label in ["Uncategorized", "Other / Unclear"]:
        st.warning(f"Cannot save '{original_filename}' with label '{label}'. Select valid category.")
        return False, None

    base_filename, _ = os.path.splitext(original_filename)
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50] or "image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    sanitized_label = sanitize_foldername(label)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
    category_path.mkdir(parents=True, exist_ok=True)

    file_extension = ".jpg" # Default
    try:
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'): file_extension = ".png"
        elif image_bytes.startswith(b'\xff\xd8\xff'): file_extension = ".jpg"
        elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP': file_extension = ".webp"
    except Exception: pass

    filename = f"{base_filename_sanitized}_{timestamp}{file_extension}"
    filepath = category_path / filename
    counter = 1
    while filepath.exists():
        filename = f"{base_filename_sanitized}_{timestamp}_{counter}{file_extension}"
        filepath = category_path / filename
        counter += 1

    try:
        with open(filepath, "wb") as f: f.write(image_bytes)
        return True, str(filepath)
    except Exception as e:
        st.error(f"Error saving to '{filepath}': {e}")
        return False, None

def get_categories_from_folders():
    ensure_library_dir()
    try:
        return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError: return []

def get_images_in_category(category_name):
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                image_files.append(item)
    return sorted(image_files, key=os.path.getmtime, reverse=True)

def delete_image_file(image_path_str):
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file():
            file_path.unlink()
            st.toast(f"Deleted: {file_path.name}", icon="🗑️")
            return True
        else:
            st.error(f"Not found for deletion: {file_path.name}")
            return False
    except Exception as e:
        st.error(f"Error deleting {image_path_str}: {e}")
        return False

def create_zip_of_library():
    ensure_library_dir()
    zip_buffer = io.BytesIO()
    added_files_count = 0
    library_path = pathlib.Path(LIBRARY_DIR)
    if not any(library_path.iterdir()): return None, 0

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for category_folder in library_path.iterdir():
            if category_folder.is_dir() and not category_folder.name.startswith('.'):
                for item in category_folder.iterdir():
                      if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                          try:
                              arcname = f"{category_folder.name}/{item.name}"
                              zipf.write(item, arcname=arcname)
                              added_files_count += 1
                          except Exception as zip_err:
                              st.warning(f"Could not add {item.name} to zip: {zip_err}")
    if added_files_count == 0: return None, 0
    zip_buffer.seek(0)
    return zip_buffer, added_files_count

def create_zip_from_folder(category_name):
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    zip_buffer = io.BytesIO()
    added_files = 0
    if not category_path.is_dir(): return None

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in category_path.iterdir():
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                try:
                    zipf.write(item, arcname=item.name)
                    added_files += 1
                except Exception as zip_err:
                    st.warning(f"Zip error for {item.name} in {category_name}: {zip_err}")
    if added_files == 0: return None
    zip_buffer.seek(0)
    return zip_buffer

# ---------- OpenAI API Setup ----------
def setup_openai_client():
    api_key = None
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API key:", type="password", key="api_key_input_sidebar", help="Required for analysis and generation.")

    if not api_key: return None
    try:
        client = OpenAI(api_key=api_key)
        # client.models.list() # Optional check
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key.")
        return None

# ---------- Utility Function ----------
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# ---------- OpenAI Analysis, Classification & ELEMENT DESCRIPTION Function ----------
def analyze_classify_and_describe_elements(client: OpenAI, image_bytes: bytes):
    """
    Analyzes thumbnail for classification AND identifies key visual elements/layout.

    Returns:
        tuple: (label, reason, layout_desc, element_descs)
        - label (str): The single most relevant category label.
        - reason (str): Status message for classification.
        - layout_desc (str | None): Description of the layout structure.
        - element_descs (list[str] | None): List of descriptions for key graphical elements.
    """
    if not client:
        return "Uncategorized", "OpenAI client not initialized.", None, None

    base64_image = encode_image(image_bytes)
    mime_type = "image/jpeg" # Default
    if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'): mime_type = "image/png"
    elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP': mime_type = "image/webp"
    image_data_uri = f"data:{mime_type};base64,{base64_image}"

    category_definitions_list = [f"{cat['name']}: {cat['description']}" for cat in STANDARD_CATEGORIES_WITH_DESC]
    category_definitions_text = "\n".join([f"- {cat_def}" for cat_def in category_definitions_list])
    valid_categories = set(STANDARD_CATEGORIES)

    system_prompt = (
        "You are an expert analyst of YouTube thumbnail visuals. Perform two tasks based on the user's request and the image:\n"
        "1.  **Classification:** Identify the single most relevant visual style category using ONLY the provided definitions. Respond with ONLY the category name.\n"
        "2.  **Element/Layout Description:** Identify the main layout structure and key distinct graphical elements (icons, logos, illustrations, borders, arrows). Describe them concisely.\n"
        "Structure your response ONLY as follows:\n"
        "CLASSIFICATION:\n[Single Category Name]\n"
        "LAYOUT:\n[Concise layout description, e.g., 'Split screen - vertical divide', 'Centered subject with background blur']\n"
        "ELEMENTS:\n"
        "- [Description of element 1, e.g., 'Bright yellow lightning bolt icon']\n"
        "- [Description of element 2, e.g., 'Circular progress bar graphic']\n"
        "(List up to 5 key distinct graphical elements. If none, state 'None found'.)"
    )
    user_prompt_text = (
       f"Analyze the thumbnail. First, classify using ONLY these definitions:\n{category_definitions_text}\n\n"
       "Second, describe layout and key graphical elements. Follow the required output format."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_text},
                        {"type": "image_url", "image_url": {"url": image_data_uri, "detail": "low"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=300
        )
        full_result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return "Uncategorized", f"Analysis failed: {e}", None, None

    label = "Uncategorized"
    reason = "Analysis complete, parsing results."
    layout_desc = None
    element_descs = []

    try:
        # Use regex for more robust parsing
        class_match = re.search(r"CLASSIFICATION:\s*(.*?)\s*(?=\nLAYOUT:|\nELEMENTS:|$)", full_result, re.DOTALL | re.IGNORECASE)
        layout_match = re.search(r"LAYOUT:\s*(.*?)\s*(?=\nELEMENTS:|$)", full_result, re.DOTALL | re.IGNORECASE)
        elements_match = re.search(r"ELEMENTS:\s*(.*)", full_result, re.DOTALL | re.IGNORECASE)

        if class_match:
            class_block = class_match.group(1).strip()
            if class_block:
                found_cat = False
                for valid_cat in valid_categories:
                    if valid_cat.strip().lower() == class_block.lower():
                        label = valid_cat
                        found_cat = True
                        break
                if not found_cat:
                    st.warning(f"AI classification unrecognized: '{class_block}'. Defaulting.")
                    label = "Other / Unclear" if "Other / Unclear" in valid_categories else "Uncategorized"
            else: st.warning("AI did not provide classification.")

        if layout_match:
            layout_block = layout_match.group(1).strip()
            if layout_block and layout_block.lower() not in ['none', 'n/a']:
                layout_desc = layout_block

        if elements_match:
            elements_block = elements_match.group(1).strip()
            if elements_block and elements_block.lower() not in ['none found', 'none', 'n/a']:
                potential_elements = elements_block.split('\n')
                for elem in potential_elements:
                    cleaned_elem = re.sub(r'^\s*[-*]\s*', '', elem).strip()
                    if cleaned_elem: element_descs.append(cleaned_elem)

        if not layout_desc and not element_descs: reason += " (No layout/elements described)."
        elif not element_descs: reason += " (No elements described)."
        elif not layout_desc: reason += " (No layout described)."

    except Exception as parse_error:
        st.error(f"Error parsing AI analysis: {parse_error}\nRaw:\n{full_result}")
        reason = f"Failed to parse AI response: {parse_error}"
        # Return potentially partial results
        return label, reason, layout_desc, element_descs

    return label, reason, layout_desc, element_descs

# ---------- Function to Analyze Category Style --------
def analyze_category_style(client: OpenAI, category_name: str, sample_size: int = STYLE_ANALYSIS_SAMPLE_SIZE):
    """Analyzes sample images from a category to describe its common visual style."""
    if not client: return None, "OpenAI client not initialized."

    image_files = get_images_in_category(category_name)
    if not image_files: return None, f"No images found in '{category_name}'."

    sample_files = random.sample(image_files, min(len(image_files), sample_size))
    if not sample_files: return None, f"Could not select samples from '{category_name}'."

    st.write(f"Analyzing style based on {len(sample_files)} sample(s) from '{category_name}'...")
    image_messages = []
    try:
        for img_path in sample_files:
            with open(img_path, "rb") as f: img_bytes = f.read()
            base64_image = encode_image(img_bytes)
            mime_type = "image/jpeg"
            if img_bytes.startswith(b'\x89PNG\r\n\x1a\n'): mime_type = "image/png"
            elif img_bytes.startswith(b'RIFF') and img_bytes[8:12] == b'WEBP': mime_type = "image/webp"
            image_messages.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "low"}})
    except Exception as e: return None, f"Error reading/encoding samples: {e}"

    analysis_prompt_text = (
        "Analyze common visual elements/style patterns across these thumbnails from the same category.\nFocus on:\n"
        "- **Layout:** (e.g., split-screen, centered subject, text position)\n"
        "- **Color Palette:** (e.g., high contrast, muted tones, dominant colors)\n"
        "- **Typography Style:** (if prominent: e.g., bold sans-serif, minimal text)\n"
        "- **Key Visual Elements:** (e.g., faces, products, graphics, icons)\n"
        "- **Overall Mood:** (e.g., energetic, serious, clean)\n\n"
        "Provide a concise description (2-3 sentences) summarizing the dominant, recurring style characteristics suitable for guiding generation of a *new* thumbnail in this style. Describe the *shared* style, not individual images."
    )
    messages = [
        {"role": "system", "content": "You are an expert visual style analyst for YouTube thumbnails."},
        {"role": "user", "content": [{"type": "text", "text": analysis_prompt_text}] + image_messages}
    ]

    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.3, max_tokens=150)
        style_description = response.choices[0].message.content.strip()
        if not style_description: return None, "Style analysis returned empty description."
        return style_description, "Style analysis successful."
    except Exception as e:
        st.error(f"Error during category style analysis: {e}")
        return None, f"Style analysis failed: {e}"

# ---------- Callbacks ----------
def add_to_library_callback(file_id, image_bytes, label, filename):
    success, saved_path = save_image_to_category(image_bytes, label, filename)
    if success:
        if 'upload_cache' in st.session_state and file_id in st.session_state.upload_cache:
            st.session_state.upload_cache[file_id]['status'] = 'added'
        if file_id.startswith("gen_"):
            st.session_state.generated_image_saved = True
        st.toast(f"Image saved to '{label}' folder!", icon="✅")
    else: st.toast(f"Failed to save image to '{label}'.", icon="❌")
    st.rerun()

def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success:
        st.session_state[f'direct_added_{file_id}'] = True
        st.toast(f"Image added to '{selected_category}' folder!", icon="⬆️")
        st.rerun()
    else: st.toast(f"Failed to add image to '{selected_category}'.", icon="❌")

def analyze_all_callback():
    if 'upload_cache' in st.session_state:
        triggered_count = 0
        for file_id, item_data in st.session_state.upload_cache.items():
            if isinstance(item_data, dict) and item_data.get('status') == 'uploaded':
                st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                triggered_count += 1
        if triggered_count > 0: st.toast(f"Triggered analysis for {triggered_count} thumbnail(s).", icon="🧠")
        else: st.toast("No thumbnails awaiting analysis.", icon="🤷")
        # No rerun here, main loop handles 'analyzing'

# ---------- Upload and Process Function (Modified for Element Selection) ----------
def upload_and_process(client: OpenAI):
    st.header("1. Upload & Analyze Thumbnails")
    st.info("Upload images. Click '🧠 Analyze All Pending' to classify and extract element descriptions. Descriptions will appear below each thumbnail.")

    # Initialize states if they don't exist
    if 'upload_cache' not in st.session_state: st.session_state.upload_cache = {}
    if 'selected_elements' not in st.session_state: st.session_state.selected_elements = {}

    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    # Process newly uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                try:
                    image_bytes = uploaded_file.getvalue()
                    # Basic validation - Ensure Pillow can open it
                    with Image.open(io.BytesIO(image_bytes)) as img: img.verify()
                    # Keep original bytes for display/potentially analysis
                    original_image_bytes = image_bytes
                    # Create processed version (e.g., consistent JPEG) for saving if needed
                    # Note: Using original bytes for analysis might yield better element detection
                    processed_image_bytes = original_image_bytes # Default to original for now

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': original_image_bytes,
                        'processed_bytes': processed_image_bytes, # Use if needed for saving standardization
                        'label': None, 'reason': "Awaiting analysis",
                        'layout': None, 'elements': [], 'status': 'uploaded'
                    }
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}. File skipped.")
                    st.session_state.upload_cache[file_id] = {
                        'status': 'error', 'error_msg': str(e), 'name': uploaded_file.name
                    }
        # st.rerun() # Optional: Rerun to show newly uploaded items immediately

    # Display and Process items from Cache
    if st.session_state.upload_cache:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            items_to_analyze = any(item.get('status') == 'uploaded' for item in st.session_state.upload_cache.values() if isinstance(item, dict))
            analyze_all_disabled = not items_to_analyze or not client
            st.button("🧠 Analyze All Pending", key="analyze_all", on_click=analyze_all_callback, disabled=analyze_all_disabled, use_container_width=True, help="Analyze classification, layout, and elements.")
        with col2:
            if st.button("Clear Uploads and Analyses", key="clear_uploads", use_container_width=True):
                st.session_state.upload_cache = {}
                st.session_state.selected_elements = {}
                st.rerun()

        st.markdown("---")
        st.header("2. Generate Using Elements")
        st.info("After analyzing, select elements via checkboxes below. Then describe the main content and generate.")

        generation_content_prompt = st.text_area("**Main Content Prompt:**", height=100, key="element_generation_content_prompt", placeholder="Describe the central theme...")
        generate_button_placeholder = st.empty() # Button will be placed after the grid

        st.markdown("---")
        st.subheader("Analyzed Thumbnails & Selectable Elements")

        num_columns = 3
        cols = st.columns(num_columns)
        col_index = 0
        any_elements_available = False

        # Use list to prevent mutation errors during iteration if items are deleted/changed
        cache_keys = list(st.session_state.upload_cache.keys())

        for file_id in cache_keys:
            item_data = st.session_state.upload_cache.get(file_id)
            if not isinstance(item_data, dict) or 'status' not in item_data: continue

            with cols[col_index % num_columns]:
                # Use CSS class for styling container
                st.markdown('<div class="thumbnail-analysis-container">', unsafe_allow_html=True)
                try:
                    if item_data['status'] == 'error':
                        st.error(f"Error: {item_data.get('error_msg', 'Unknown')}")
                        st.caption(f"File: {item_data.get('name', 'Unknown')}")
                    else:
                        st.image(item_data['original_bytes'], caption=item_data.get('name', 'Unnamed'), use_container_width=True)
                        analysis_placeholder = st.empty()

                        if item_data['status'] == 'uploaded':
                            analysis_placeholder.info("Ready for analysis.")
                        elif item_data['status'] == 'analyzing':
                            with analysis_placeholder.container():
                                with st.spinner(f"Analyzing {item_data['name']}..."):
                                    # Use original bytes for analysis
                                    label, reason, layout, elements = analyze_classify_and_describe_elements(client, item_data['original_bytes'])
                                    if file_id in st.session_state.upload_cache: # Check if item still exists
                                        st.session_state.upload_cache[file_id].update({
                                            'label': label, 'reason': reason, 'layout': layout,
                                            'elements': elements or [], 'status': 'analyzed'
                                        })
                                        st.rerun() # Update display after analysis
                        elif item_data['status'] in ['analyzed', 'added']:
                            with analysis_placeholder.container():
                                st.markdown(f"**Class:** `{item_data.get('label', 'N/A')}`")
                                layout = item_data.get('layout')
                                st.markdown(f"**Layout:** _{layout}_" if layout else "**Layout:** _Not described_")

                                elements = item_data.get('elements', [])
                                if elements:
                                    st.markdown("**Select Elements:**")
                                    any_elements_available = True
                                    for i, desc in enumerate(elements):
                                        element_key = f"elem_{file_id}_{i}"
                                        # Use session state to manage checkbox state persistently
                                        is_selected = st.checkbox(
                                            f"{desc}",
                                            value=(element_key in st.session_state.selected_elements), # Set initial value from state
                                            key=element_key,
                                            help=f"Select this element: {desc}",
                                            # Use CSS class for styling checkbox label if needed
                                            #label_markdown_unsafe=f'<span class="element-checkbox">{desc}</span>'
                                        )
                                        # Update selected_elements state based on checkbox interaction
                                        if is_selected:
                                            st.session_state.selected_elements[element_key] = desc
                                        elif element_key in st.session_state.selected_elements:
                                            del st.session_state.selected_elements[element_key]
                                else:
                                    st.caption("Elements: None described")
                except KeyError as e:
                    st.error(f"Missing data for item {file_id}: {e}")
                except Exception as e:
                    st.error(f"Display error for {item_data.get('name', file_id)}: {e}")
                finally:
                    st.markdown('</div>', unsafe_allow_html=True) # Close container div
            col_index += 1

        # --- Place the Generate Button ---
        selected_element_descriptions = list(st.session_state.selected_elements.values())
        num_selected = len(selected_element_descriptions)
        generate_disabled = not generation_content_prompt or num_selected == 0 or not client
        button_text = f"🎨 Generate with Selected Elements ({num_selected})" if num_selected > 0 else "🎨 Select Elements to Generate"

        with generate_button_placeholder.container():
            st.markdown("---")
            if st.button(button_text, key="generate_with_elements_button", disabled=generate_disabled, use_container_width=True):
                st.session_state.generated_image_b64 = None
                st.session_state.generation_prompt_used = None
                st.session_state.generated_image_saved = False
                st.session_state.generation_error = None
                with st.spinner("Generating thumbnail using selected elements..."):
                    try:
                        element_list_str = "\n".join([f"- {desc}" for desc in selected_element_descriptions])
                        full_prompt = (
                            f"Create a YouTube thumbnail image (16:9 aspect ratio).\n\n"
                            f"**Main Content:**\n{generation_content_prompt}\n\n"
                            f"**Required Elements (incorporate based on descriptions):**\n{element_list_str}\n\n"
                            f"**Instructions:**\n"
                            f"- Arrange main content and required elements clearly and visually appealingly.\n"
                            f"- Ensure elements are recognizable based on descriptions.\n"
                            f"- Maintain a style suitable for a high-clickthrough YouTube thumbnail (bright, clear, engaging).\n"
                            f"- If multiple elements requested, try for stylistic consistency.\n"
                            f"- Avoid extra text unless part of main content prompt."
                        )
                        st.caption("Sending prompt to DALL-E 3...")
                        # print(f"DALL-E Prompt (Elements): {full_prompt}") # Debug

                        response = client.images.generate(model="dall-e-3", prompt=full_prompt, size="1792x1024", quality="hd", n=1, response_format="b64_json")

                        if response.data and response.data[0].b64_json:
                            b64_data = response.data[0].b64_json
                            st.session_state.generated_image_b64 = b64_data
                            prompt_hint = f"{generation_content_prompt[:20]}_{num_selected}_elements"
                            st.session_state.generation_prompt_used = prompt_hint
                            st.session_state.generated_image_saved = False
                            st.session_state.generation_error = None
                        else:
                            st.error("❌ Generation OK but no image data received.")
                            st.session_state.generation_error = "API success but no image data."
                    except Exception as e:
                        error_message = f"❌ Thumbnail generation failed: {e}"
                        st.error(error_message)
                        if hasattr(e, 'response') and e.response: st.error(f"API Response: {e.response.text}")
                        if "safety system" in str(e).lower(): st.warning("Prompt possibly blocked by safety system.")
                        st.session_state.generation_error = error_message
                st.rerun() # Rerun to show results/errors in the display section below

    elif not uploaded_files:
        st.markdown("<p style='text-align: center; font-style: italic;'>Upload thumbnails to analyze elements!</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("3. Generation Result")
    display_generated_image_section() # Shared display for results

# ---------- Library Explorer ----------
def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category. Delete images or download category Zips.")

    if 'confirm_delete_path' not in st.session_state: st.session_state.confirm_delete_path = None
    if "selected_category_folder" not in st.session_state: st.session_state.selected_category_folder = None

    categories = get_categories_from_folders()

    if st.session_state.confirm_delete_path:
        display_delete_confirmation()
        return # Stop rendering explorer while confirming delete

    if st.session_state.selected_category_folder is None:
        st.markdown("### Select a Category Folder")
        if not categories:
            st.info("Library empty. Add images via 'Upload & Analyze' or 'Generate'.")
            return

        cols_per_row = 5
        num_rows = (len(categories) + cols_per_row - 1) // cols_per_row
        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < len(categories):
                    cat_name = categories[idx]
                    if cols[j].button(cat_name, key=f"btn_lib_{cat_name}", use_container_width=True):
                        st.session_state.selected_category_folder = cat_name
                        st.rerun()
    else:
        selected_category = st.session_state.selected_category_folder
        st.markdown(f"### Category: **{selected_category}**")

        top_cols = st.columns([0.2, 0.5, 0.3]) # Back, Add Direct, Download Zip
        with top_cols[0]:
            if st.button("⬅️ Back", key="back_button", use_container_width=True):
                st.session_state.selected_category_folder = None
                st.session_state.confirm_delete_path = None
                st.rerun()
        with top_cols[1]:
            with st.expander(f"⬆️ Add Image to '{selected_category}'"):
                direct_uploaded_file = st.file_uploader(f"Upload for '{selected_category}'", type=["jpg", "jpeg", "png", "webp"], key=f"direct_upload_{selected_category}", label_visibility="collapsed")
                if direct_uploaded_file:
                    file_id = f"direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}"
                    if f'direct_added_{file_id}' not in st.session_state: st.session_state[f'direct_added_{file_id}'] = False
                    is_added = st.session_state[f'direct_added_{file_id}']
                    st.image(direct_uploaded_file, width=150)
                    try:
                        img_bytes_direct = direct_uploaded_file.getvalue()
                        with Image.open(io.BytesIO(img_bytes_direct)) as img_direct:
                            img_direct.verify() # Verify it's a valid image
                            # Re-open after verify
                            with Image.open(io.BytesIO(img_bytes_direct)) as img_to_process:
                                if img_to_process.mode in ['RGBA', 'P']: img_to_process = img_to_process.convert("RGB")
                                img_byte_arr_direct = io.BytesIO()
                                img_to_process.save(img_byte_arr_direct, format='JPEG', quality=85)
                                processed_bytes_direct = img_byte_arr_direct.getvalue()

                        st.button(f"⬆️ Add This Image" if not is_added else "✔️ Added", key=f"btn_direct_add_{file_id}", on_click=add_direct_to_library_callback, args=(file_id, processed_bytes_direct, selected_category, direct_uploaded_file.name), disabled=is_added, use_container_width=True)
                    except Exception as e: st.error(f"Failed to process direct upload: {e}")

        image_files = get_images_in_category(selected_category)
        with top_cols[2]:
             if image_files:
                 zip_buffer = create_zip_from_folder(selected_category)
                 if zip_buffer:
                     st.download_button(label=f"⬇️ Download ({len(image_files)})", data=zip_buffer, file_name=f"{sanitize_foldername(selected_category)}_thumbnails.zip", mime="application/zip", key=f"download_{selected_category}", use_container_width=True)
                 else: st.button(f"⬇️ Download ({len(image_files)})", disabled=True, use_container_width=True, help="No valid images.")
             else: st.button("⬇️ Download (0)", disabled=True, use_container_width=True)

        if image_files:
            st.markdown("---")
            cols_per_row_thumbs = 5
            thumb_cols = st.columns(cols_per_row_thumbs)
            col_idx = 0
            for image_path in image_files:
                with thumb_cols[col_idx % cols_per_row_thumbs]:
                    st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True) # Use CSS class
                    try:
                        image_path_str = str(image_path)
                        st.image(image_path_str, caption=f"{image_path.name}", use_container_width=True)
                        st.markdown('<div class="delete-button-container">', unsafe_allow_html=True) # Wrap button
                        mtime = image_path.stat().st_mtime
                        del_key = f"del_{image_path.name}_{mtime}"
                        if st.button("🗑️", key=del_key, help="Delete this image"):
                            st.session_state.confirm_delete_path = image_path_str
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True) # Close button container
                    except Exception as img_err: st.warning(f"Load error: {image_path.name} ({img_err})")
                    finally: st.markdown('</div>', unsafe_allow_html=True) # Close item container
                col_idx += 1
        elif not direct_uploaded_file: # Only show if not empty and not currently uploading
            st.info(f"No thumbnails found in '{selected_category}'. Add via Upload or the expander above.")

# ---------- Delete Confirmation Dialog Function ----------
def display_delete_confirmation():
    file_to_delete = st.session_state.confirm_delete_path
    if not file_to_delete: return
    with st.warning(f"**Confirm Deletion:** Delete `{os.path.basename(file_to_delete)}` permanently?"):
        col1, col2, _ = st.columns([1.5, 1, 5])
        with col1:
            if st.button("🔥 Confirm Delete", key="confirm_delete_yes", type="primary"):
                delete_image_file(file_to_delete)
                st.session_state.confirm_delete_path = None
                st.rerun()
        with col2:
            if st.button("🚫 Cancel", key="confirm_delete_cancel"):
                st.session_state.confirm_delete_path = None
                st.rerun()

# ---------- Direct Thumbnail Generation UI ----------
def thumbnail_generator_direct_ui(client: OpenAI):
    st.header("Generate Thumbnail Directly")
    st.markdown("Describe the thumbnail, select optional *standard* style categories, and generate.")
    if not client: st.error("❌ OpenAI client needed. Provide API key in sidebar."); return

    prompt_text = st.text_area("**Thumbnail Content Prompt:**", height=150, key="generator_direct_prompt_text", placeholder="e.g., Futuristic cityscape sunset...")
    category_map = {cat['name']: cat['description'] for cat in STANDARD_CATEGORIES_WITH_DESC}
    style_options = STANDARD_CATEGORIES
    selected_style_categories = st.multiselect("**Reference Style Categories (Optional - Standard Definitions):**", options=style_options, key="generator_direct_style_categories", help="Guide style based on general definitions.")

    if st.button("✨ Generate Directly", key="generate_thumb_direct", disabled=not prompt_text):
        st.session_state.generated_image_b64 = None; st.session_state.generation_prompt_used = None
        st.session_state.generated_image_saved = False; st.session_state.generation_error = None
        with st.spinner("Generating with DALL-E 3..."):
            try:
                full_prompt = f"Create hyper-realistic YouTube thumbnail (16:9) depicting: {prompt_text}."
                if selected_style_categories:
                    style_hints = [f"- Style '{name}': {category_map[name]}" for name in selected_style_categories if name in category_map]
                    if style_hints: full_prompt += "\n\nIncorporate stylistic elements based on standard definitions:\n" + "\n".join(style_hints)
                else: full_prompt += "\n\nUse visually striking style for high clickthrough."
                full_prompt += "\n\nEnsure high quality, clear, follows prompt. Avoid complex/unreadable text unless requested."
                st.caption("Sending prompt to DALL-E 3...")
                # print(f"DALL-E Prompt (Direct): {full_prompt}") # Debug

                response = client.images.generate(model="dall-e-3", prompt=full_prompt, size="1792x1024", quality="hd", n=1, response_format="b64_json")

                if response.data and response.data[0].b64_json:
                    st.session_state.generated_image_b64 = response.data[0].b64_json
                    st.session_state.generation_prompt_used = prompt_text
                    st.session_state.generated_image_saved = False; st.session_state.generation_error = None
                else:
                    st.error("❌ Generation OK but no image data.")
                    st.session_state.generation_error = "API success but no image data."
            except Exception as e:
                error_message = f"❌ Generation failed: {e}"
                st.error(error_message)
                if hasattr(e, 'response') and e.response: st.error(f"API Response: {e.response.text}")
                if "safety system" in str(e).lower(): st.warning("Prompt possibly blocked by safety system.")
                st.session_state.generation_error = error_message
        st.rerun() # Rerun to show result/error

    display_generated_image_section() # Shared display

# ---------- Generate by Category Style UI ----------
def generate_by_category_style_ui(client: OpenAI):
    st.header("Generate by Category Style")
    st.markdown("Select library category, describe content. AI analyzes category style & combines.")
    if not client: st.error("❌ OpenAI client needed. Provide API key in sidebar."); return

    available_categories = get_categories_from_folders()
    style_ref_categories = [cat for cat in available_categories if cat not in ["uncategorized", "Other / Unclear"]]
    if not style_ref_categories: st.warning("Need categories with images in library first."); return

    selected_category = st.selectbox("**Select Reference Category:**", options=style_ref_categories, key="generator_style_ref_category", help="AI will analyze this category's style.")
    content_prompt = st.text_area("**Thumbnail Content Prompt:**", height=150, key="generator_style_content_prompt", placeholder="Describe main subject/scene...")

    generate_disabled = not selected_category or not content_prompt
    if st.button("🎨 Generate with Category Style", key="generate_thumb_style", disabled=generate_disabled):
        st.session_state.generated_image_b64 = None; st.session_state.generation_prompt_used = None
        st.session_state.generated_image_saved = False; st.session_state.generation_error = None
        style_desc, analysis_msg = None, ""
        with st.spinner(f"Analyzing style of '{selected_category}'..."):
            style_desc, analysis_msg = analyze_category_style(client, selected_category)

        if not style_desc:
            st.error(f"Failed style analysis: {analysis_msg}")
            st.session_state.generation_error = f"Style analysis failed: {analysis_msg}"
            st.rerun(); return # Stop and show error via rerun

        st.success(f"Style analysis complete for '{selected_category}'.")
        with st.expander("Derived Style Description (for AI)"): st.caption(style_desc)

        with st.spinner("Generating thumbnail using analyzed style..."):
            try:
                full_prompt = (
                    f"Create YouTube thumbnail (16:9) depicting: {content_prompt}.\n\n"
                    f"IMPORTANT: Visual style MUST strongly adhere to these characteristics derived from similar thumbnails:\n"
                    f"--- Style Description Start ---\n{style_desc}\n--- Style Description End ---\n\n"
                    f"Ensure high quality, clear, integrates content naturally within style. Avoid text unless in content prompt."
                )
                st.caption("Sending combined prompt to DALL-E 3...")
                # print(f"DALL-E Prompt (Styled): {full_prompt}") # Debug

                response = client.images.generate(model="dall-e-3", prompt=full_prompt, size="1792x1024", quality="hd", n=1, response_format="b64_json")

                if response.data and response.data[0].b64_json:
                    st.session_state.generated_image_b64 = response.data[0].b64_json
                    st.session_state.generation_prompt_used = content_prompt # Use content prompt for filename hint
                    st.session_state.generated_image_saved = False; st.session_state.generation_error = None
                else:
                    st.error("❌ Generation OK but no image data.")
                    st.session_state.generation_error = "API success but no image data."
            except Exception as e:
                error_message = f"❌ Generation failed: {e}"
                st.error(error_message)
                if hasattr(e, 'response') and e.response: st.error(f"API Response: {e.response.text}")
                if "safety system" in str(e).lower(): st.warning("Prompt possibly blocked by safety system.")
                st.session_state.generation_error = error_message
        st.rerun() # Rerun to show result/error

    display_generated_image_section() # Shared display

# ---------- Shared Function to Display Generated Image & Save Options --------
def display_generated_image_section():
    """ Displays the generated image (if available) and options to save it. """
    # Display error first if it occurred
    if st.session_state.get("generation_error"):
        st.error(f"Previous generation attempt failed: {st.session_state.generation_error}")

    if 'generated_image_b64' in st.session_state and st.session_state.generated_image_b64:
        # st.subheader("Generated Thumbnail Result") # Headers now placed in calling functions
        is_saved = st.session_state.get('generated_image_saved', False)
        st.markdown('<div class="generated-image-container">', unsafe_allow_html=True)
        try:
            generated_image_bytes = base64.b64decode(st.session_state.generated_image_b64)
            st.image(generated_image_bytes, caption="Generated Image Preview", use_container_width=True)
            st.markdown("---")
            st.markdown("**Save Generated Thumbnail to Library:**")

            current_folders = get_categories_from_folders()
            savable_categories = [cat for cat in current_folders if cat not in ["uncategorized", "other_unclear"]]
            if not savable_categories:
                st.warning("No suitable library categories found. Create a category folder first.")
                save_category = None; save_button_disabled = True
            else:
                save_category = st.selectbox("Choose destination category:", options=savable_categories, key="save_gen_category_select")
                save_button_disabled = not save_category or is_saved

            prompt_part = re.sub(r'\W+', '_', st.session_state.get('generation_prompt_used', 'generated'))[:40].strip('_') or "generated"
            generated_filename = f"{prompt_part}.png" # DALL-E b64 is PNG
            gen_file_id = f"gen_{prompt_part[:10]}_{int(time.time())}" # Simple unique ID

            st.button(
                "✅ Save to Selected Category" if not is_saved else "✔️ Saved",
                key=f"btn_save_gen_{gen_file_id}",
                on_click=add_to_library_callback,
                args=(gen_file_id, generated_image_bytes, save_category, generated_filename),
                disabled=save_button_disabled,
                use_container_width=True,
                help="Save to chosen category." if not save_button_disabled else ("Already saved" if is_saved else "Select category")
            )
        except Exception as display_err:
            st.error(f"Error displaying/saving generated image: {display_err}")
            st.session_state.generated_image_b64 = None # Clear broken state
        finally: st.markdown('</div>', unsafe_allow_html=True)

# ---------- Main App ----------
def main():
    ensure_library_dir()
    create_predefined_category_folders(STANDARD_CATEGORIES)

    # Initialize Session State Keys robustly
    keys_to_init = {
        'selected_category_folder': None, 'upload_cache': {}, 'confirm_delete_path': None,
        'generated_image_b64': None, 'generation_prompt_used': None, 'generated_image_saved': False,
        'generation_error': None, 'nav_menu': None, 'selected_elements': {}, '_previous_nav_menu': None
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state: st.session_state[key] = default_value

    # --- Sidebar Setup ---
    with st.sidebar:
        st.markdown('<div style="text-align: center; padding: 10px;">🖼️</div>', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center;">Thumbnail Toolkit</h2>', unsafe_allow_html=True)
        st.markdown("Analyze, organize, generate.", unsafe_allow_html=True)
        st.markdown("---")

        client = setup_openai_client()
        if not client: st.warning("OpenAI API key needed for Analysis/Generation.", icon="🔑")

        # --- Navigation ---
        menu_options = ["Upload & Analyze", "Generate Directly", "Generate by Category Style", "Library Explorer"]
        default_menu_option = menu_options[0]
        # Ensure current selection is valid
        current_selection = st.session_state.get("nav_menu")
        if current_selection not in menu_options:
             st.session_state.nav_menu = default_menu_option # Reset to default if invalid
             current_selection = default_menu_option

        menu_index = menu_options.index(current_selection)
        # Use st.session_state.nav_menu directly with st.radio's key parameter for persistence
        st.radio("Navigation", menu_options, key="nav_menu", index=menu_index, label_visibility="collapsed")
        menu = st.session_state.nav_menu # Get the selection AFTER the radio button renders

        st.markdown("---")
        st.info(f"Library: `./{LIBRARY_DIR}`")
        try:
            zip_buffer, file_count = create_zip_of_library()
            if zip_buffer and file_count > 0:
                 st.download_button(label=f"⬇️ Download All ({file_count})", data=zip_buffer, file_name="thumbnail_library.zip", mime="application/zip", key="download_all", use_container_width=True)
            else: st.button("⬇️ Download All (Empty)", disabled=True, use_container_width=True)
        except Exception as zip_all_err:
              st.error(f"Zip Error: {zip_all_err}")
              st.button("⬇️ Download All (Error)", disabled=True, use_container_width=True)
        st.markdown("---")
        with st.expander("Standard Category Definitions"):
             for cat in STANDARD_CATEGORIES_WITH_DESC:
                 if cat['name'] != "Other / Unclear": st.markdown(f"**{cat['name']}**: _{cat['description']}_")

    # --- Main Content Area ---
    if st.session_state.confirm_delete_path:
        display_delete_confirmation()
    else:
        # Clear generation state if navigating away from generator tabs
        active_generator_tabs = ["Generate Directly", "Generate by Category Style", "Upload & Analyze"]
        previous_menu = st.session_state.get('_previous_nav_menu', None)
        if previous_menu != menu: # Check if menu selection actually changed
             if previous_menu in active_generator_tabs and menu not in active_generator_tabs:
                 if st.session_state.get('generated_image_b64') is not None:
                     # Clear state ONLY if navigating AWAY from ALL generator tabs
                     st.session_state.generated_image_b64 = None; st.session_state.generation_prompt_used = None
                     st.session_state.generated_image_saved = False; st.session_state.generation_error = None
                     st.session_state.selected_elements = {} # Clear element selections too
        # Update tracker AFTER potentially clearing state
        st.session_state['_previous_nav_menu'] = menu

        # Render selected page
        if menu == "Upload & Analyze": upload_and_process(client)
        elif menu == "Generate Directly": thumbnail_generator_direct_ui(client)
        elif menu == "Generate by Category Style": generate_by_category_style_ui(client)
        elif menu == "Library Explorer": library_explorer()

if __name__ == "__main__":
    main()
