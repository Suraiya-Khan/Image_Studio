import streamlit as st
import cv2
import numpy as np

# --- Helper Functions ---
def resize_image(img, width, height):
    return cv2.resize(img, (width, height))

def scale_image(img, fx, fy):
    return cv2.resize(img, None, fx=fx, fy=fy)

def crop_rectangle(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def flip_image(img, flip_code):
    return cv2.flip(img, flip_code)

def convert_image(img, option):
    if option == "Grayscale":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif option == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia = cv2.transform(img, kernel)
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif option == "Negative":
        return cv2.bitwise_not(img)
    else:
        return img

# --- Streamlit UI ---
st.set_page_config(page_title="Image Processing Studio", page_icon="🖼️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>✨ Image Processing Studio ✨</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image and choose an operation from the sidebar</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Original Image", width=600)

    # Sidebar menu
    st.sidebar.title("⚙️ Options")
    option = st.sidebar.radio("Choose Operation", ["Resize", "Crop", "Flip", "Convert"])

    processed_img = None  # store result

    # --- Resize ---
    if option == "Resize":
        st.sidebar.subheader("Resize Settings")
        new_width = st.sidebar.number_input("Width", value=600, min_value=50)
        new_height = st.sidebar.number_input("Height", value=600, min_value=50)
        fx = st.sidebar.slider("Scale X (fx)", 0.1, 2.0, 0.2)
        fy = st.sidebar.slider("Scale Y (fy)", 0.1, 2.0, 0.2)

        if st.sidebar.button("Apply Fixed Resize"):
            processed_img = resize_image(img, new_width, new_height)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Resized Image", width=600)

        if st.sidebar.button("Apply Scaling"):
            processed_img = scale_image(img, fx, fy)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Scaled Image", width=600)

    # --- Crop (Rectangle) ---
    elif option == "Crop":
        st.sidebar.subheader("Crop Settings (Rectangle)")
        h, w = img.shape[:2]

        x = st.sidebar.number_input("X Position", min_value=0, max_value=w-1, value=int(w*0.25))
        y = st.sidebar.number_input("Y Position", min_value=0, max_value=h-1, value=int(h*0.25))
        rect_w = st.sidebar.number_input("Crop Width", min_value=10, max_value=w-x, value=int(w*0.5))
        rect_h = st.sidebar.number_input("Crop Height", min_value=10, max_value=h-y, value=int(h*0.5))

        if st.sidebar.button("Apply Crop"):
            processed_img = crop_rectangle(img, x, y, rect_w, rect_h)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Cropped Image", width=600)

    # --- Flip ---
    elif option == "Flip":
        st.sidebar.subheader("Flip Settings")
        flip_option = st.sidebar.selectbox("Flip Type", ["Horizontal", "Vertical", "Both"])
        flip_code = 1 if flip_option == "Horizontal" else 0 if flip_option == "Vertical" else -1
        if st.sidebar.button("Apply Flip"):
            processed_img = flip_image(img, flip_code)
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"{flip_option} Flip", width=600)

    # --- Convert ---
    elif option == "Convert":
        st.sidebar.subheader("Convert Settings")
        convert_choice = st.sidebar.selectbox("Conversion Type", ["Grayscale", "Sepia", "Negative"])
        if st.sidebar.button("Apply Conversion"):
            processed_img = convert_image(img, convert_choice)
            if convert_choice == "Grayscale":
                st.image(processed_img, caption="Grayscale Image", width=600)
            else:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"{convert_choice} Image", width=600)

    # --- Download Button ---
    if processed_img is not None:
        success, buffer = cv2.imencode(".png", processed_img)
        if success:
            st.download_button(
                label="💾 Download Processed Image",
                data=buffer.tobytes(),
                file_name="processed_image.png",
                mime="image/png"
            )
