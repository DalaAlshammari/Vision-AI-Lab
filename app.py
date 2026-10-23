import io

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# Page configuration
st.set_page_config(
    page_title="Vision AI Lab",
    page_icon="👁️",
    layout="wide"
)


# Application title
st.title("👁️ Vision AI Lab")
st.write("AI-powered object detection using computer vision.")
st.divider()


# Load the pretrained YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")


# Sidebar settings
with st.sidebar:
    st.header("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.40,
        step=0.05
    )

    st.caption(
        "Higher values show only more confident detections."
    )


# Image uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:
    try:
        # Read uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        # Run object detection
        with st.spinner("Detecting objects..."):
            model = load_model()

            results = model.predict(
                source=image,
                conf=confidence,
                verbose=False
            )

            result = results[0]

            # Generate annotated image
            detected_image = result.plot()
            detected_image = detected_image[:, :, ::-1]

        with right:
            st.subheader("Detection Results")
            st.image(
                detected_image,
                use_container_width=True
            )

        # Detection summary
        st.subheader("📊 Detection Summary")

        boxes = result.boxes
        total = len(boxes)

        metric1, metric2 = st.columns(2)

        metric1.metric("Objects Detected", total)
        metric2.metric(
            "Confidence Threshold",
            f"{confidence:.0%}"
        )

        # Display detected objects
        if total > 0:
            detections = []

            for box in boxes:
                class_id = int(box.cls[0].item())
                label = result.names[class_id]
                score = float(box.conf[0].item())

                detections.append({
                    "Object": label,
                    "Confidence": f"{score:.1%}"
                })

            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No objects detected at this confidence threshold."
            )

        # Prepare image for download
        buffer = io.BytesIO()

        Image.fromarray(detected_image).convert("RGB").save(
            buffer,
            format="PNG"
        )

        # Download annotated image
        st.download_button(
            "⬇️ Download Annotated Image",
            data=buffer.getvalue(),
            file_name="detection_result.png",
            mime="image/png"
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Upload an image to start detecting objects.")

    st.caption(
        "The model can detect common objects such as people, cars, "
        "bottles, chairs, and animals."
    )