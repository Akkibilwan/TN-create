import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re
import pathlib
import time
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env (if available)

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library"

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
    /* Existing CSS */
    .thumbnail-container, .db-thumbnail-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .analysis-box {
        margin-top: 10px;
        padding: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        border: 1px solid #ced4da;
    }
    .analysis-box p {
       margin-bottom: 5px;
       font-size: 0.9em;
    }
    .delete-button-container {
        margin-top: 10px;
        text-align: center;
    }
    .delete-button-container button {
       width: 80%;
       background-color: #dc3545;
       color: white;
       border: none;
       padding: 5px 10px;
       border-radius: 4px;
       cursor: pointer;
       transition: background-color 0.2s ease;
    }
    .delete-button-container button:hover {
        background-color: #c82333;
    }
    .stButton>button {
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stDownloadButton>button {
        width: 100%;
    }
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
    try:
        pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        st.error(f"Error creating library directory: {e}")


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
        st.warning(f"Cannot save image '{original_filename}' with label '{label}'. Please select a valid category.")
        return False, None

    try:
        base_filename, _ = os.path.splitext(original_filename)
        base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50]
        if not base_filename_sanitized:
            base_filename_sanitized = "image"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

        sanitized_label = sanitize_foldername(label)
        category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
        category_path.mkdir(parents=True, exist_ok=True)

        file_extension = ".jpg"
        try:
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                file_extension = ".png"
            elif image_bytes.startswith(b'\xff\xd8\xff'):
                file_extension = ".jpg"
            elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
                file_extension = ".webp"
        except Exception:
            pass

        filename = f"{base_filename_sanitized}_{timestamp}{file_extension}"
        filepath = category_path / filename
        counter = 1
        while filepath.exists():
            filename = f"{base_filename_sanitized}_{timestamp}_{counter}{file_extension}"
            filepath = category_path / filename
            counter += 1

        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return True, str(filepath)

    except Exception as e:
        st.error(f"Error saving image to '{filepath}': {e}")
        return False, None


def get_categories_from_folders():
    ensure_library_dir()
    try:
        return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error getting categories: {e}")
        return []


def get_images_in_category(category_name):
    try:
        sanitized_category = sanitize_foldername(category_name)
        category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
        image_files = []
        if category_path.is_dir():
            for item in category_path.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png',
                                                             '.webp'] and not item.name.startswith('.'):
                    image_files.append(item)
        return sorted(image_files, key=os.path.getmtime, reverse=True)
    except Exception as e:
        st.error(f"Error getting images in category: {e}")
        return []


def delete_image_file(image_path_str):
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file():
            file_path.unlink()
            st.toast(f"Deleted: {file_path.name}", icon="🗑️")
            return True
        else:
            st.error(f"File not found for deletion: {file_path.name}")
            return False
    except Exception as e:
        st.error(f"Error deleting file {image_path_str}: {e}")
        return False


def create_zip_of_library():
    ensure_library_dir()
    zip_buffer = io.BytesIO()
    added_files_count = 0
    library_path = pathlib.Path(LIBRARY_DIR)

    if not any(library_path.iterdir()):
        return None, 0

    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for category_folder in library_path.iterdir():
                if category_folder.is_dir() and not category_folder.name.startswith('.'):
                    for item in category_folder.iterdir():
                        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png',
                                                                     '.webp'] and not item.name.startswith('.'):
                            try:
                                arcname = f"{category_folder.name}/{item.name}"
                                zipf.write(item, arcname=arcname)
                                added_files_count += 1
                            except Exception as zip_err:
                                st.warning(f"Could not add {item.name} to zip: {zip_err}")

        if added_files_count == 0:
            return None, 0

        zip_buffer.seek(0)
        return zip_buffer, added_files_count

    except Exception as e:
        st.error(f"Error creating zip of library: {e}")
        return None, 0


# ---------- OpenAI API Setup ----------

def setup_openai_client():
    api_key = None
    try:
        if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            api_key = st.sidebar.text_input(
                "Enter OpenAI API key:",
                type="password",
                key="api_key_input_sidebar",
                help="Required for analyzing and generating thumbnails."
            )

        if not api_key:
            st.sidebar.warning("OpenAI API key is missing.")
            return None

        client = OpenAI(api_key=api_key)
        # client.models.list()  # Optional: Test call
        return client

    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key.")
        return None


# ---------- Utility Function ----------

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')


# ---------- OpenAI Analysis & Classification Function (Updated Categories) ----------

def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    try:
        base64_image = encode_image(image_bytes)
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"

        category_definitions_list = [f"{cat['name']}: {cat['description']}" for cat in
                                     STANDARD_CATEGORIES_WITH_DESC]
        category_definitions_text = "\n".join([f"- {cat_def}" for cat_def in category_definitions_list])

        valid_categories = set(STANDARD_CATEGORIES)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert analyst of YouTube thumbnail visual styles. Analyze the provided image and identify the **single most relevant** visual style category using ONLY the following definitions. Respond ONLY with the single category name from the list. Do NOT include numbers, prefixes like 'Label:', reasoning, or explanation."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Classify this thumbnail using ONLY these definitions, providing the single most relevant category name:\n{category_definitions_text}\n\nOutput ONLY the single category name."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_uri, "detail": "low"}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip()

        label = "Uncategorized"
        reason = "Analysis complete."

        if result:
            found = False
            for valid_cat in valid_categories:
                if valid_cat.strip().lower() == result.strip().lower():
                    label = valid_cat
                    found = True
                    break
            if not found:
                st.warning(
                    f"AI returned unrecognized category: '{result}'. Classifying as 'Other / Unclear'.")
                label = "Other / Unclear" if "Other / Unclear" in valid_categories else "Uncategorized"
        else:
            st.warning("AI returned an empty category response. Classifying as 'Uncategorized'.")
            label = "Uncategorized"

        return label, reason

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return "Uncategorized", f"Analysis failed: {e}"


# ---------- Callbacks ----------

def add_to_library_callback(file_id, image_bytes, label, filename):
    try:
        success, saved_path = save_image_to_category(image_bytes, label, filename)
        if success:
            if 'upload_cache' in st.session_state and file_id in st.session_state.upload_cache:
                st.session_state.upload_cache[file_id]['status'] = 'added'

            if file_id.startswith("gen_"):
                st.session_state.generated_image_saved = True

            st.toast(f"Image saved to '{label}' folder!", icon="✅")
        else:
            st.toast(f"Failed to save image to '{label}'.", icon="❌")
        st.rerun()

    except Exception as e:
        st.error(f"Error in add_to_library_callback: {e}")


def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    try:
        success, _ = save_image_to_category(image_bytes, selected_category, filename)
        if success:
            st.session_state[f'direct_added_{file_id}'] = True
            st.toast(f"Image added to '{selected_category}' folder!", icon="⬆️")
            st.rerun()
        else:
            st.toast(f"Failed to add image directly to '{selected_category}'.", icon="❌")
    except Exception as e:
        st.error(f"Error in add_direct_to_library_callback: {e}")


def analyze_all_callback():
    try:
        if 'upload_cache' in st.session_state:
            triggered_count = 0
            for file_id, item_data in st.session_state.upload_cache.items():
                if isinstance(item_data, dict) and item_data.get('status') == 'uploaded':
                    st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                    triggered_count += 1
            if triggered_count > 0:
                st.toast(f"Triggered analysis for {triggered_count} thumbnail(s).", icon="🧠")
            else:
                st.toast("No thumbnails awaiting analysis.", icon="🤷")
    except Exception as e:
        st.error(f"Error in analyze_all_callback: {e}")


# ---------- Upload and Process Function ----------

def upload_and_process(client: OpenAI):
    st.header("Upload & Analyze Thumbnails")
    st.info(
        "Upload images, click '🧠 Analyze All Pending', then '✅ Add to Library' to save to the suggested category folder.")

    if 'upload_cache' not in st.session_state:
        st.session_state.upload_cache = {}

    uploaded_files = st.file_uploader(
        "Choose thumbnail images...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                try:
                    image_bytes = uploaded_file.getvalue()
                    display_image = Image.open(io.BytesIO(image_bytes))
                    display_image.verify()
                    display_image = Image.open(io.BytesIO(image_bytes))

                    img_byte_arr = io.BytesIO()
                    if display_image.mode == 'RGBA' or display_image.mode == 'P':
                        processed_image = display_image.convert('RGB')
                    else:
                        processed_image = display_image

                    processed_image.save(img_byte_arr, format='JPEG', quality=85)
                    processed_image_bytes = img_byte_arr.getvalue()

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': image_bytes,
                        'processed_bytes': processed_image_bytes,
                        'label': None,
                        'reason': "Awaiting analysis",
                        'status': 'uploaded'
                    }
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}. File skipped.")
                    st.session_state.upload_cache[file_id] = {
                        'status': 'error',
                        'error_msg': str(e),
                        'name': uploaded_file.name
                    }

    if st.session_state.upload_cache:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            items_to_analyze = any(
                isinstance(item, dict) and item.get('status') == 'uploaded'
                for item in st.session_state.upload_cache.values()
            )
            analyze_all_disabled = not items_to_analyze or not client
            st.button(
                "🧠 Analyze All Pending",
                key="analyze_all",
                on_click=analyze_all_callback,
                disabled=analyze_all_disabled,
                use_container_width=True,
                help="Requires OpenAI API Key" if not client else "Analyze all thumbnails not yet processed"
            )

        with col2:
            if st.button("Clear Uploads and Analyses", key="clear_uploads", use_container_width=True):
                st.session_state.upload_cache = {}
                st.rerun()

        st.markdown("---")

        num_columns = 4
        cols = st.columns(num_columns)
        col_index = 0

        for file_id in list(st.session_state.upload_cache.keys()):
            item_data = st.session_state.upload_cache.get(file_id)

            if not isinstance(item_data, dict) or 'status' not in item_data:
                continue

            with cols[col_index % num_columns]:
                with st.container(border=False):
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    try:
                        if item_data['status'] == 'error':
                            st.error(
                                f"Error with {item_data.get('name', 'Unknown File')}: {item_data.get('error_msg', 'Unknown error')}")
                        else:
                            st.image(
                                item_data['original_bytes'],
                                caption=f"{item_data.get('name', 'Unnamed Thumbnail')}",
                                use_container_width=True
                            )

                            analysis_placeholder = st.empty()

                            if item_data['status'] == 'uploaded':
                                analysis_placeholder.info("Ready for analysis.")
                            elif item_data['status'] == 'analyzing':
                                with analysis_placeholder.container():
                                    with st.spinner(f"Analyzing {item_data['name']}..."):
                                        label, reason = analyze_and_classify_thumbnail(client,
                                                                                       item_data['processed_bytes'])
                                        if file_id in st.session_state.upload_cache:
                                            st.session_state.upload_cache[file_id]['label'] = label
                                            st.session_state.upload_cache[file_id]['reason'] = reason
                                            st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                            st.rerun()
                            elif item_data['status'] in ['analyzed', 'added']:
                                with analysis_placeholder.container():
                                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                    label = item_data.get('label', 'Uncategorized')
                                    st.markdown(f"**Suggested:** `{label}`")

                                    is_added = (item_data['status'] == 'added')
                                    add_button_disabled = (
                                        is_added or
                                        label == "Uncategorized" or
                                        not label or
                                        label == "Other / Unclear"
                                    )

                                    st.button(
                                        "✅ Add to Library" if not is_added else "✔️ Added",
                                        key=f'btn_add_{file_id}',
                                        on_click=add_to_library_callback,
                                        args=(file_id, item_data['processed_bytes'], label, item_data['name']),
                                        disabled=add_button_disabled,
                                        use_container_width=True,
                                        help="Save this image to the suggested category folder." if not add_button_disabled else (
                                            "Image already added" if is_added else "Cannot add Uncategorized or Other/Unclear images")
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)

                    except KeyError as e:
                        st.error(f"Missing data for an item: {e}. Try re-uploading.")
                        if file_id in st.session_state.upload_cache:
                            del st.session_state.upload_cache[file_id]
                    except Exception as e:
                        st.error(f"Display error for {item_data.get('name', file_id)}: {e}")
                    finally:
                        st.markdown('</div>', unsafe_allow_html=True)

            col_index += 1

    elif not uploaded_files:
        st.markdown("<p style='text-align: center; font-style: italic;'>Upload some thumbnails to get started!</p>",
                    unsafe_allow_html=True)


# ---------- Function to create Zip File from Single Folder ----------

def create_zip_from_folder(category_name):
    try:
        sanitized_category = sanitize_foldername(category_name)
        category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
        zip_buffer = io.BytesIO()
        added_files = 0

        if not category_path.is_dir():
            return None

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in category_path.iterdir():
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png',
                                                             '.webp'] and not item.name.startswith('.'):
                    try:
                        zipf.write(item, arcname=item.name)
                        added_files += 1
                    except Exception as zip_err:
                        st.warning(f"Zip error for {item.name} in {category_name}: {zip_err}")

        if added_files == 0:
            return None

        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"Error creating zip from folder: {e}")
        return None


# ---------- Library Explorer ----------

def library_explorer():
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category folder. Delete images or download category Zips.")

    if 'confirm_delete_path' not in st.session_state:
        st.session_state.confirm_delete_path = None
    if "selected_category_folder" not in st.session_state:
        st.session_state.selected_category_folder = None

    categories = get_categories_from_folders()

    if st.session_state.confirm_delete_path:
        display_delete_confirmation()
        return

    if st.session_state.selected_category_folder is None:
        st.markdown("### Select a Category Folder to View")
        if not categories:
            st.info(
                "Your thumbnail library is currently empty. Add some images via the 'Upload & Analyze' tab or 'Generate Thumbnail' tab.")
            return

        cols_per_row = 5
        num_categories = len(categories)
        num_rows = (num_categories + cols_per_row - 1) // cols_per_row

        for i in range(num_rows):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i * cols_per_row + j
                if idx < num_categories:
                    cat_name = categories[idx]
                    if cols[j].button(cat_name, key=f"btn_lib_{cat_name}", use_container_width=True):
                        st.session_state.selected_category_folder = cat_name
                        st.rerun()

    else:
        selected_category = st.session_state.selected_category_folder
        st.markdown(f"### Category Folder: **{selected_category}**")

        top_cols = st.columns([0.2, 0.5, 0.3])

        with top_cols[0]:
            if st.button("⬅️ Back", key="back_button", use_container_width=True,
                         help="Go back to category list"):
                st.session_state.selected_category_folder = None
                st.session_state.confirm_delete_path = None
                st.rerun()

        with top_cols[1]:
            with st.expander(f"⬆️ Add Image Directly to '{selected_category}'"):
                direct_uploaded_file = st.file_uploader(
                    f"Upload image for '{selected_category}'",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"direct_upload_{selected_category}",
                    label_visibility="collapsed"
                )
                if direct_uploaded_file:
                    file_id = f"direct_{selected_category}_{direct_uploaded_file.name}_{direct_uploaded_file.size}"

                    if f'direct_added_{file_id}' not in st.session_state:
                        st.session_state[f'direct_added_{file_id}'] = False

                    is_added = st.session_state[f'direct_added_{file_id}']

                    st.image(direct_uploaded_file, width=150)

                    try:
                        img_bytes_direct = direct_uploaded_file.getvalue()
                        img_direct = Image.open(io.BytesIO(img_bytes_direct))
                        img_direct.verify()
                        img_direct = Image.open(io.BytesIO(img_bytes_direct))

                        if img_direct.mode == 'RGBA' or img_direct.mode == 'P':
                            img_direct = img_direct.convert("RGB")

                        img_byte_arr_direct = io.BytesIO()
                        img_direct.save(img_byte_arr_direct, format='JPEG', quality=85)
                        processed_bytes_direct = img_byte_arr_direct.getvalue()

                        st.button(
                            f"⬆️ Add This Image" if not is_added else "✔️ Added",
                            key=f"btn_direct_add_{file_id}",
                            on_click=add_direct_to_library_callback,
                            args=(file_id, processed_bytes_direct, selected_category,
                                  direct_uploaded_file.name),
                            disabled=is_added,
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Failed to process direct upload: {e}")

        image_files = get_images_in_category(selected_category)

        if image_files:
            with top_cols[2]:
                zip_buffer = create_zip_from_folder(selected_category)
                if zip_buffer:
                    st.download_button(
                        label=f"⬇️ Download ({len(image_files)})",
                        data=zip_buffer,
                        file_name=f"{sanitize_foldername(selected_category)}_thumbnails.zip",
                        mime="application/zip",
                        key=f"download_{selected_category}",
                        use_container_width=True,
                        help=f"Download all images in '{selected_category}' as a zip file."
                    )
                else:
                    st.button(f"⬇️ Download ({len(image_files)})", disabled=True,
                              use_container_width=True, help="No valid images to download.")

        st.markdown("---")
        cols_per_row_thumbs = 5
        thumb_cols = st.columns(cols_per_row_thumbs)
        col_idx = 0
        for image_path in image_files:
            with thumb_cols[col_idx % cols_per_row_thumbs]:
                with st.container(border=False):
                    st.markdown('<div class="db-thumbnail-container">', unsafe_allow_html=True)
                    try:
                        image_path_str = str(image_path)
                        st.image(image_path_str, caption=f"{image_path.name}",
                                 use_container_width=True)

                        st.markdown('<div class="delete-button-container">',
                                    unsafe_allow_html=True)
                        mtime = image_path.stat().st_mtime
                        del_key = f"del_{image_path.name}_{mtime}"

                        if st.button("🗑️", key=del_key, help="Delete this image"):
                            st.session_state.confirm_delete_path = image_path_str
                            st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as img_err:
                        st.warning(f"Could not load: {image_path.name} ({img_err})")
                    finally:
                        st.markdown('</div>', unsafe_allow_html=True)
            col_idx += 1

        elif not direct_uploaded_file or (
                direct_uploaded_file and not st.session_
