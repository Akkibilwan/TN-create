import streamlit as st
import os
import io
import base64
import zipfile
from datetime import datetime
from PIL import Image
from openai import OpenAI
import re
import pathlib  # For path manipulation
# import shutil # Not strictly needed as zipfile handles folder zipping
import time
import requests  # Needed for downloading generated image if URL is returned (though using b64)

# --- Configuration ---
LIBRARY_DIR = "thumbnail_library"  # Main directory for storing category folders

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
    page_title="Thumbnail Toolkit (Analyze & Generate)",  # Updated Title
    page_icon="🖼️",  # Changed icon
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
        position: relative; /* Needed for absolute positioning of delete button */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 100%; /* Make containers equal height in a row */
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Push button to bottom */
    }
    .analysis-box {
        margin-top: 10px;
        padding: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        border: 1px solid #ced4da;
    }
    .analysis-box p { /* Style paragraphs inside analysis box */
       margin-bottom: 5px;
       font-size: 0.9em;
    }
    .delete-button-container {
        /* Removed absolute positioning, button flows naturally */
        margin-top: 10px; /* Add space above the button */
        text-align: center; /* Center the button */
    }
    .delete-button-container button {
       width: 80%; /* Make delete button slightly smaller */
       background-color: #dc3545; /* Red color */
       color: white;
       border: none;
       padding: 5px 10px;
       border-radius: 4px;
       cursor: pointer;
       transition: background-color 0.2s ease;
    }
    .delete-button-container button:hover {
        background-color: #c82333; /* Darker red on hover */
    }
    .stButton>button { /* Style general Streamlit buttons */
        border-radius: 5px;
        padding: 8px 15px;
    }
    .stDownloadButton>button { /* Style download buttons */
        width: 100%;
    }
     /* Ensure generated image fits well */
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
    # (Sanitize function remains the same)
    name = name.strip()
    # Remove or replace characters invalid in Windows/Linux/macOS folder names
    # Including .,; which might cause issues in some contexts
    name = re.sub(r'[<>:"/\\|?*.,;]+', '_', name)
    # Replace multiple consecutive underscores with a single one
    name = re.sub(r'_+', '_', name)
    # Handle reserved names in Windows
    if name.upper() in ["CON", "PRN", "AUX", "NUL"] or re.match(r"^(COM|LPT)[1-9]$", name.upper()):
        name = f"_{name}_"
    return name if name else "uncategorized"


def ensure_library_dir():
    # (Ensure dir function remains the same)
    pathlib.Path(LIBRARY_DIR).mkdir(parents=True, exist_ok=True)


# Function to Pre-create Standard Category Folders
def create_predefined_category_folders(category_list):
    """Creates folders for standard categories if they don't exist."""
    ensure_library_dir()
    # st.sidebar.write("Checking standard category folders...") # Less verbose
    created_count = 0
    for category_name in category_list:  # Now iterates through names only
        sanitized_name = sanitize_foldername(category_name)
        # Avoid creating folders for generic/empty names unless explicitly desired
        if not sanitized_name or sanitized_name in ["uncategorized", "other_unclear"]:
            continue

        folder_path = pathlib.Path(LIBRARY_DIR) / sanitized_name
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                created_count += 1
            except Exception as e:
                st.sidebar.warning(f"Could not create folder for '{category_name}': {e}")
    # Show message only if folders were created
    if created_count > 0:
        st.sidebar.caption(f"Created {created_count} new category folders.")


# Modified for single label, handles bytes input
def save_image_to_category(image_bytes, label, original_filename="thumbnail"):
    """Saves image bytes to the specified category folder."""
    ensure_library_dir()
    if not label or label in ["Uncategorized", "Other / Unclear"]:
        st.warning(f"Cannot save image '{original_filename}' with label '{label}'. Please select a valid category.")
        return False, None

    # Sanitize base filename more aggressively
    base_filename, _ = os.path.splitext(original_filename)
    base_filename_sanitized = re.sub(r'[^\w\-]+', '_', base_filename).strip('_')[:50]  # Max 50 chars
    if not base_filename_sanitized:
        base_filename_sanitized = "image"  # Fallback if name becomes empty

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # microseconds added

    sanitized_label = sanitize_foldername(label)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_label
    category_path.mkdir(parents=True, exist_ok=True)  # Create just in case

    # Determine file extension (try to keep original if known, default to jpg)
    # For generated images (b64 from DALL-E is usually PNG) or processed uploads (we convert to JPG)
    file_extension = ".jpg"  # Default for processed uploads
    try:
        # Check common magic numbers
        if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            file_extension = ".png"
        elif image_bytes.startswith(b'\xff\xd8\xff'):
            file_extension = ".jpg"
        elif image_bytes.startswith(b'RIFF') and image_bytes[8:12] == b'WEBP':
            file_extension = ".webp"
        # Add more checks if needed (GIF, etc.)
    except Exception:
        pass  # Keep default if check fails or bytes too short

    # If original filename had a known extension, maybe prioritize it?
    # _, orig_ext = os.path.splitext(original_filename)
    # if orig_ext.lower() in ['.png', '.webp', '.jpeg', '.jpg']:
    #     file_extension = orig_ext.lower() # Consider using original extension

    filename = f"{base_filename_sanitized}_{timestamp}{file_extension}"
    filepath = category_path / filename
    counter = 1
    while filepath.exists():
        # Append counter if filename collision
        filename = f"{base_filename_sanitized}_{timestamp}_{counter}{file_extension}"
        filepath = category_path / filename
        counter += 1

    try:
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return True, str(filepath)
    except Exception as e:
        st.error(f"Error saving image to '{filepath}': {e}")
        return False, None


def get_categories_from_folders():
    # (Function remains the same)
    ensure_library_dir()
    try:
        # List directories, filter out hidden ones (like .DS_Store)
        return sorted([d.name for d in pathlib.Path(LIBRARY_DIR).iterdir() if d.is_dir() and not d.name.startswith('.')])
    except FileNotFoundError:
        return []


def get_images_in_category(category_name):
    # (Function remains the same)
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    image_files = []
    if category_path.is_dir():
        for item in category_path.iterdir():
            # Check for common image extensions, ignore hidden files
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp'] and not item.name.startswith('.'):
                image_files.append(item)
    # Sort by modification time, newest first
    return sorted(image_files, key=os.path.getmtime, reverse=True)


def delete_image_file(image_path_str):
    # (Function remains the same)
    try:
        file_path = pathlib.Path(image_path_str)
        if file_path.is_file():
            file_path.unlink()  # Delete the file
            st.toast(f"Deleted: {file_path.name}", icon="🗑️")
            return True
        else:
            st.error(f"File not found for deletion: {file_path.name}")
            return False
    except Exception as e:
        st.error(f"Error deleting file {image_path_str}: {e}")
        return False


# ---------- NEW: Function to create Zip File of Entire Library ----------
def create_zip_of_library():
    """Creates a zip file containing all category folders and their images."""
    ensure_library_dir()
    zip_buffer = io.BytesIO()
    added_files_count = 0
    library_path = pathlib.Path(LIBRARY_DIR)

    if not any(library_path.iterdir()):  # Check if library directory is empty
        return None, 0

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate through each category folder in the library directory
        for category_folder in library_path.iterdir():
            if category_folder.is_dir() and not category_folder.name.startswith('.'):
                # Iterate through files within the category folder
                for item in category_folder.iterdir():  # Use iterdir instead of glob('*')
                    if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png',
                                                                 '.webp'] and not item.name.startswith('.'):
                        try:
                            # Add file to zip, preserving directory structure
                            arcname = f"{category_folder.name}/{item.name}"
                            zipf.write(item, arcname=arcname)
                            added_files_count += 1
                        except Exception as zip_err:
                            st.warning(f"Could not add {item.name} to zip: {zip_err}")

    if added_files_count == 0:
        return None, 0  # No files were added

    zip_buffer.seek(0)
    return zip_buffer, added_files_count


# ---------- OpenAI API Setup ----------
def setup_openai_client():
    # (setup_openai_client remains the same)
    api_key = None
    # Try Streamlit secrets
    if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        # Try environment variables
        api_key = os.environ.get('OPENAI_API_KEY')

    # If not found, ask user in sidebar
    if not api_key:
        api_key = st.sidebar.text_input(
            "Enter OpenAI API key:",
            type="password",
            key="api_key_input_sidebar",
            help="Required for analyzing and generating thumbnails."
        )

    if not api_key:
        # st.sidebar.warning("OpenAI API key is missing.") # Show warning persistently if needed
        return None  # Return None if no key

    try:
        client = OpenAI(api_key=api_key)
        # Optional: Add a simple test call here if needed to verify key
        # client.models.list()
        return client
    except Exception as e:
        st.sidebar.error(f"Error initializing OpenAI client: {e}. Check API key.")
        return None


# ---------- Utility Function ----------
def encode_image(image_bytes):
    # (encode_image remains the same)
    return base64.b64encode(image_bytes).decode('utf-8')


# ---------- OpenAI Analysis & Classification Function (Updated Categories) ----------
def analyze_and_classify_thumbnail(client: OpenAI, image_bytes: bytes):
    """ Analyzes thumbnail for the single most relevant label from the expanded list. """
    if not client:
        return "Uncategorized", "OpenAI client not initialized."

    base64_image = encode_image(image_bytes)
    image_data_uri = f"data:image/jpeg;base64,{base64_image}"  # Assuming JPEG bytes after processing

    # --- Use Category Definitions from STANDARD_CATEGORIES_WITH_DESC ---
    category_definitions_list = [f"{cat['name']}: {cat['description']}" for cat in
                                 STANDARD_CATEGORIES_WITH_DESC]
    category_definitions_text = "\n".join([f"- {cat_def}" for cat_def in category_definitions_list])

    # --- Use STANDARD_CATEGORIES list for validation ---
    valid_categories = set(STANDARD_CATEGORIES)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Use the specified model
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
                            "image_url": {"url": image_data_uri, "detail": "low"}  # Use low detail for faster analysis
                        }
                    ]
                }
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=50  # Slightly increased buffer for longer names like 'Arrow/Circle Emphasis'
        )
        result = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error during OpenAI analysis: {e}")
        return "Uncategorized", "Analysis failed due to an API error."

    # Validate the single label output
    label = "Uncategorized"  # Default
    reason = "Analysis complete."  # Simple default reason

    try:
        if result:
            found = False
            # Check against STANDARD_CATEGORIES (case-insensitive comparison)
            for valid_cat in valid_categories:
                if valid_cat.strip().lower() == result.strip().lower():
                    label = valid_cat  # Use the official casing from STANDARD_CATEGORIES
                    found = True
                    break
            if not found:
                st.warning(
                    f"AI returned unrecognized category: '{result}'. Classifying as 'Other / Unclear'.")
                # Fallback to 'Other / Unclear' if defined, else 'Uncategorized'
                label = "Other / Unclear" if "Other / Unclear" in valid_categories else "Uncategorized"
        else:
            st.warning("AI returned an empty category response. Classifying as 'Uncategorized'.")
            label = "Uncategorized"

    except Exception as parse_error:
        st.warning(
            f"Could not process AI label response: '{result}'. Error: {parse_error}. Classifying as 'Uncategorized'.")
        label = "Uncategorized"

    # The 'reason' part isn't really used or stored meaningfully anymore with the single label focus.
    return label, reason


# ---------- Callbacks ----------
def add_to_library_callback(file_id, image_bytes, label, filename):
    """Callback to save an image (uploaded or generated) to the library."""
    # Note: file_id helps manage state for uploaded items, less critical for generated ones unless tracking saved status
    success, saved_path = save_image_to_category(image_bytes, label, filename)
    if success:
        # Update status for uploaded items if applicable
        if 'upload_cache' in st.session_state and file_id in st.session_state.upload_cache:
            st.session_state.upload_cache[file_id]['status'] = 'added'

        # Set flag for generated images if applicable (check file_id prefix)
        if file_id.startswith("gen_"):
            st.session_state.generated_image_saved = True  # Mark as saved

        st.toast(f"Image saved to '{label}' folder!", icon="✅")
    else:
        st.toast(f"Failed to save image to '{label}'.", icon="❌")

    # Rerun to update button states (e.g., "Added", "Saved") and potentially clear generated image section if desired
    # For generated images, setting the flag might be enough, rerun might clear the image display. Let's test.
    # A rerun is generally needed to reflect the 'added' status in the upload list correctly.
    # Let's keep the rerun for now.
    st.rerun()


def add_direct_to_library_callback(file_id, image_bytes, selected_category, filename):
    # (Callback remains largely the same)
    success, _ = save_image_to_category(image_bytes, selected_category, filename)
    if success:
        # Use a unique key to track added status for direct uploads
        st.session_state[f'direct_added_{file_id}'] = True
        st.toast(f"Image added to '{selected_category}' folder!", icon="⬆️")
        st.rerun()  # Rerun to update the button state in the expander
    else:
        st.toast(f"Failed to add image directly to '{selected_category}'.", icon="❌")


def analyze_all_callback():
    # (Callback remains the same)
    if 'upload_cache' in st.session_state:
        triggered_count = 0
        for file_id, item_data in st.session_state.upload_cache.items():
            # Only trigger analysis if status is 'uploaded'
            if isinstance(item_data, dict) and item_data.get('status') == 'uploaded':
                st.session_state.upload_cache[file_id]['status'] = 'analyzing'
                triggered_count += 1
        if triggered_count > 0:
            st.toast(f"Triggered analysis for {triggered_count} thumbnail(s).", icon="🧠")
            # No rerun here needed, the main loop will see 'analyzing' status and handle it
        else:
            st.toast("No thumbnails awaiting analysis.", icon="🤷")


# ---------- Upload and Process Function ----------
def upload_and_process(client: OpenAI):
    # (Function logic is largely the same, uses updated analyze function)
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

    # Process newly uploaded files
    if uploaded_files:
        # new_files_added = False # Removed rerun based on this
        for uploaded_file in uploaded_files:
            # Create a unique ID based on name and size
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_id not in st.session_state.upload_cache:
                # new_files_added = True
                try:
                    image_bytes = uploaded_file.getvalue()
                    # Basic validation and conversion to JPEG bytes for consistency
                    display_image = Image.open(io.BytesIO(image_bytes))
                    display_image.verify()  # Verify image data
                    # Re-open after verify
                    display_image = Image.open(io.BytesIO(image_bytes))

                    # Convert to RGB (for JPEG saving) and save to bytes buffer
                    img_byte_arr = io.BytesIO()
                    # Ensure image is converted to RGB before saving as JPEG
                    if display_image.mode == 'RGBA' or display_image.mode == 'P':
                        processed_image = display_image.convert('RGB')
                    else:
                        processed_image = display_image

                    processed_image.save(img_byte_arr, format='JPEG', quality=85)  # Use quality 85
                    processed_image_bytes = img_byte_arr.getvalue()

                    st.session_state.upload_cache[file_id] = {
                        'name': uploaded_file.name,
                        'original_bytes': image_bytes,  # Keep original bytes for display
                        'processed_bytes': processed_image_bytes,  # Use processed for analysis/saving
                        'label': None,
                        'reason': "Awaiting analysis",
                        'status': 'uploaded'  # Initial status
                    }
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}. File skipped.")
                    # Add error status to cache to prevent re-processing attempts
                    st.session_state.upload_cache[file_id] = {
                        'status': 'error',
                        'error_msg': str(e),
                        'name': uploaded_file.name
                    }
        # Optional: Rerun immediately after processing uploads to show them instantly
        # if new_files_added: st.rerun()

    # Display and Process items from Cache
    if st.session_state.upload_cache:
        st.markdown("---")
        # Control Buttons
        col1, col2 = st.columns(2)
        with col1:
            # Check if any items have 'uploaded' status
            items_to_analyze = any(
                isinstance(item, dict) and item.get('status') == 'uploaded'
                for item in st.session_state.upload_cache.values()
            )
            analyze_all_disabled = not items_to_analyze or not client
            # Use the callback for 'Analyze All'
            st.button(
                "🧠 Analyze All Pending",
                key="analyze_all",
                on_click=analyze_all_callback,
                disabled=analyze_all_disabled,
                use_container_width=True,
                help="Requires OpenAI API Key" if not client else "Analyze all thumbnails not yet processed"
            )

        with col2:
            # Button to clear the cache
            if st.button("Clear Uploads and Analyses", key="clear_uploads", use_container_width=True):
                st.session_state.upload_cache = {}
                st.rerun()  # Rerun to clear the display

        st.markdown("---")

        # Thumbnail Grid
        num_columns = 4  # Adjust number of columns
        cols = st.columns(num_columns)
        col_index = 0

        # Iterate over a copy of keys to allow modification during iteration (if needed, though callbacks handle it now)
        for file_id in list(st.session_state.upload_cache.keys()):
            item_data = st.session_state.upload_cache.get(file_id)  # Use .get for safety

            # Skip if item_data is somehow invalid or removed
            if not isinstance(item_data, dict) or 'status' not in item_data:
                continue

            with cols[col_index % num_columns]:
                # Use st.container for grouping elements for each thumbnail
                with st.container(border=False):  # Use border=False if using CSS border
                    st.markdown('<div class="thumbnail-container">', unsafe_allow_html=True)
                    try:
                        if item_data['status'] == 'error':
                            st.error(
                                f"Error with {item_data.get('name', 'Unknown File')}: {item_data.get('error_msg', 'Unknown error')}")
                        else:
                            # Display image using original bytes
                            st.image(
                                item_data['original_bytes'],
                                caption=f"{item_data.get('name', 'Unnamed Thumbnail')}",
                                use_container_width=True
                            )

                            analysis_placeholder = st.empty()  # Placeholder for status/buttons

                            # Handle different statuses
                            if item_data['status'] == 'uploaded':
                                analysis_placeholder.info("Ready for analysis.")
                            elif item_data['status'] == 'analyzing':
                                # This block runs when status is set by the analyze_all_callback
                                with analysis_placeholder.container():
                                    with st.spinner(f"Analyzing {item_data['name']}..."):
                                        # Perform analysis - Use processed bytes
                                        label, reason = analyze_and_classify_thumbnail(client,
                                                                                       item_data['processed_bytes'])
                                        # Update cache entry - MUTATING session state here
                                        if file_id in st.session_state.upload_cache:  # Check if still exists
                                            st.session_state.upload_cache[file_id]['label'] = label
                                            st.session_state.upload_cache[file_id]['reason'] = reason  # Store reason/status msg
                                            st.session_state.upload_cache[file_id]['status'] = 'analyzed'
                                            # Rerun needed to update display from 'analyzing' to 'analyzed' state
                                            st.rerun()
                            elif item_data['status'] in ['analyzed', 'added']:
                                # Display results and 'Add to Library' button
                                with analysis_placeholder.container():
                                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                    label = item_data.get('label', 'Uncategorized')
                                    st.markdown(f"**Suggested:** `{label}`")

                                    is_added = (item_data['status'] == 'added')
                                    add_button_disabled = (
                                        is_added or
                                        label == "Uncategorized" or
                                        not label or
                                        label == "Other / Unclear"  # Also disable for Other/Unclear
                                    )

                                    st.button(
                                        "✅ Add to Library" if not is_added else "✔️ Added",
                                        key=f'btn_add_{file_id}',
                                        on_click=add_to_library_callback,
                                        args=(file_id, item_data['processed_bytes'], label, item_data['name']),
                                        disabled=add_button_disabled,
                                        use_container_width=True,  # Make button fill width
                                        help="Save this image to the suggested category folder." if not add_button_disabled else (
                                            "Image already added" if is_added else "Cannot add Uncategorized or Other/Unclear images")
                                    )
                                    st.markdown('</div>', unsafe_allow_html=True)

                    except KeyError as e:
                        st.error(f"Missing data for an item: {e}. Try re-uploading.")
                        if file_id in st.session_state.upload_cache:
                            del st.session_state.upload_cache[file_id]  # Clean up bad entry
                    except Exception as e:
                        st.error(f"Display error for {item_data.get('name', file_id)}: {e}")
                    finally:
                        st.markdown('</div>', unsafe_allow_html=True)  # Close container div

            col_index += 1

    # Message if cache is empty and no files were uploaded in this run
    elif not uploaded_files:
        st.markdown("<p style='text-align: center; font-style: italic;'>Upload some thumbnails to get started!</p>",
                    unsafe_allow_html=True)


# ---------- Function to create Zip File from Single Folder ----------
def create_zip_from_folder(category_name):
    # (Function remains the same)
    sanitized_category = sanitize_foldername(category_name)
    category_path = pathlib.Path(LIBRARY_DIR) / sanitized_category
    zip_buffer = io.BytesIO()
    added_files = 0

    if not category_path.is_dir():
        return None  # Category folder doesn't exist

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in category_path.iterdir():
            # Ensure it's a file, has an image extension, and not hidden
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png',
                                                         '.webp'] and not item.name.startswith('.'):
                try:
                    # Add file to the root of the zip archive
                    zipf.write(item, arcname=item.name)
                    added_files += 1
                except Exception as zip_err:
                    st.warning(f"Zip error for {item.name} in {category_name}: {zip_err}")

    if added_files == 0:
        return None  # No valid image files found in the folder

    zip_buffer.seek(0)  # Rewind buffer to the beginning
    return zip_buffer


# ---------- Library Explorer ----------
def library_explorer():
    # (Function remains largely the same, displays folders, includes delete)
    st.header("Thumbnail Library Explorer")
    st.markdown("Browse saved thumbnails by category folder. Delete images or download category Zips.")

    # Initialize state variables if they don't exist
    if 'confirm_delete_path' not in st.session_state:
        st.session_state.confirm_delete_path = None
    if "selected_category_folder" not in st.session_state:
        st.session_state.selected_category_folder = None

    # Get current categories from filesystem
    categories = get_categories_from_folders()

    # --- Display Confirmation Dialog if an image is marked for deletion ---
    # This check MUST happen before the main explorer view renders
    if st.session_state.confirm_delete_path:
        display_delete_confirmation()  # Call the confirmation dialog function
        # Stop further execution in this explorer view while confirming
        return  # IMPORTANT: Prevent rest
