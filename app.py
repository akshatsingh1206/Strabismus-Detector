# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import io
import gc
from tensorflow.keras import backend as K

st.set_page_config(page_title="Eye Strabismus Analyzer", layout="wide")

# ---------------- CONFIG -----------------
IMG_WIDTH, IMG_HEIGHT = 640, 400

# ---------------- HELPERS -----------------
def read_bytes(uploaded):
    """Return raw bytes from Streamlit UploadedFile or bytes input."""
    if uploaded is None:
        return None
    if hasattr(uploaded, "read"):
        b = uploaded.read()
        try:
            uploaded.seek(0)
        except Exception:
            pass
        return b
    return uploaded  # already bytes

def preprocess_image_bytes(image_bytes):
    """Preprocess bytes -> model input (batch, H, W, 1)"""
    if image_bytes is None:
        raise ValueError("No image bytes provided.")
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- METRICS (same as before) -----------------
def SCLERA_pixel_accuracy(y_true, y_pred):
    y_true_bin = K.cast(K.equal(y_true, 1), K.floatx())
    dynamic_threshold = K.mean(y_pred)
    threshold = tf.clip_by_value(dynamic_threshold, 0.3, 0.7)
    y_pred_bin = K.cast(K.greater(y_pred, threshold), K.floatx())
    correct_pixels = K.sum(y_true_bin * y_pred_bin, axis=[1, 2, 3])
    total_iris_pixels = K.sum(y_true_bin, axis=[1, 2, 3])
    accuracy_per_image = correct_pixels / (total_iris_pixels + K.epsilon())
    return K.mean(accuracy_per_image)

def PUPIL_pixel_accuracy(y_true, y_pred):
    y_true_bin = K.cast(K.equal(y_true, 1), K.floatx())
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    correct_pixels = K.sum(y_true_bin * y_pred_bin)
    total_iris_pixels = K.sum(y_true_bin)
    return correct_pixels / (total_iris_pixels + K.epsilon())

@st.cache_resource
def load_models():
    """Load models once and reuse (cache_resource for Streamlit)."""
    # Make sure files exist in the app folder: iris_ggg.h5 and pupilggg.h5
    modelS = tf.keras.models.load_model(
        "iris_ggg.h5",
        custom_objects={'iris_pixel_accuracy': SCLERA_pixel_accuracy},
        compile=False
    )
    modelP = tf.keras.models.load_model(
        "pupilggg.h5",
        custom_objects={'iris_pixel_accuracy': PUPIL_pixel_accuracy},
        compile=False
    )
    return modelS, modelP

def predict_mask_from_bytes(model, image_bytes, threshold=0.4):
    image = preprocess_image_bytes(image_bytes)
    pred = model.predict(image)[0, :, :, 0]
    mask = (pred > threshold).astype(np.uint8) * 255
    return mask

# ---------------- GEOMETRY/PROCESSING -----------------
def safe_find_contours(mask):
    """Return contours or empty list if none found."""
    if mask is None:
        return []
    mask_u8 = (mask.copy()).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours is not None else []

def get_sclera_points(mask):
    contours = safe_find_contours(mask)
    if not contours:
        return (0, 0), (0, 0), (0, 0), np.array([[[0, 0]]])
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    pts = contour[:, 0, :]
    left = tuple(pts[np.argmin(pts[:, 0])])
    right = tuple(pts[np.argmax(pts[:, 0])])
    return (cx, cy), left, right, contour

def get_pupil_center(mask):
    contours = safe_find_contours(mask)
    if not contours:
        return (0, 0)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) >= 5:
        (cx, cy), _, _ = cv2.fitEllipse(contour)
        return (cx, cy)
    M = cv2.moments(contour)
    cx = (M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    cy = (M["m01"] / M["m00"]) if M["m00"] != 0 else 0
    return (cx, cy)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.dot(ba, bc) / denom
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle_rad)

def process_eye(sclera_mask, pupil_mask):
    sclera_center, left_edge, right_edge, _ = get_sclera_points(sclera_mask)
    pupil_center = get_pupil_center(pupil_mask)
    eye_length = euclidean(left_edge, right_edge)
    return {
        "sclera_center": sclera_center,
        "left_edge": left_edge,
        "right_edge": right_edge,
        "pupil_center": pupil_center,
        "eye_length": eye_length,
        "distances": {
            "pupil_to_sclera": euclidean(pupil_center, sclera_center),
            "pupil_to_left": euclidean(pupil_center, left_edge),
            "pupil_to_right": euclidean(pupil_center, right_edge),
        },
        "angles": {
            "eye_span": angle_between(left_edge, sclera_center, right_edge),
            "gaze_angle": angle_between(sclera_center, pupil_center, (sclera_center[0] + 1, sclera_center[1])),
        }
    }

# ---------------- UI -----------------
st.title("👁️ Eye Strabismus Analyzer")
st.write("Upload *left* and *right* eye images. You can drag & drop or use camera input (if available).")

col_left, col_right = st.columns(2)

with col_left:
    left_file = st.file_uploader("Left eye image", type=["png", "jpg", "jpeg"], key="left")
    left_cam = st.camera_input("Or take left eye photo (camera)", key="cam_left")
    # prefer camera input if provided
    left_bytes = read_bytes(left_cam) if left_cam is not None else read_bytes(left_file)
    if left_bytes is not None:
        st.image(left_bytes, caption="Left preview", use_column_width=True)
    else:
        st.info("Left eye image preview will appear here after upload or camera photo.")

with col_right:
    right_file = st.file_uploader("Right eye image", type=["png", "jpg", "jpeg"], key="right")
    right_cam = st.camera_input("Or take right eye photo (camera)", key="cam_right")
    right_bytes = read_bytes(right_cam) if right_cam is not None else read_bytes(right_file)
    if right_bytes is not None:
        st.image(right_bytes, caption="Right preview", use_column_width=True)
    else:
        st.info("Right eye image preview will appear here after upload or camera photo.")

# Controls
col_btns1, col_btns2, _ = st.columns([1, 1, 2])
with col_btns1:
    analyze = st.button("🔍 Analyze")
with col_btns2:
    clear = st.button("Reset")

if clear:
    st.experimental_rerun()

if analyze:
    if left_bytes is None or right_bytes is None:
        st.error("Please provide both left and right images before analyzing.")
    else:
        with st.spinner("Loading models and analyzing..."):
            try:
                modelS, modelP = load_models()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                raise

            # predict masks
            try:
                sclera_left = predict_mask_from_bytes(modelS, left_bytes, threshold=0.4)
                sclera_right = predict_mask_from_bytes(modelS, right_bytes, threshold=0.4)
                pupil_left = predict_mask_from_bytes(modelP, left_bytes, threshold=0.1)
                pupil_right = predict_mask_from_bytes(modelP, right_bytes, threshold=0.1)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                raise

            # process geometry and metrics
            left_eye = process_eye(sclera_left, pupil_left)
            right_eye = process_eye(sclera_right, pupil_right)

            # prepare metrics text
            def compare_metric(label, lv, rv, base, threshold_percent):
                diff = abs(lv - rv)
                threshold = base * threshold_percent
                ok = diff <= threshold
                return {
                    "label": label, "left": lv, "right": rv, "diff": diff, "threshold": threshold, "ok": ok
                }

            base_len = (left_eye["eye_length"] + right_eye["eye_length"]) / 2 if (left_eye["eye_length"] + right_eye["eye_length"])>0 else 1
            m1 = compare_metric("Pupil vs Sclera Center",
                                left_eye["distances"]["pupil_to_sclera"], right_eye["distances"]["pupil_to_sclera"],
                                base_len, 0.08)
            m2 = compare_metric("Pupil vs Left Edge",
                                left_eye["distances"]["pupil_to_left"], right_eye["distances"]["pupil_to_left"],
                                base_len, 0.08)
            m3 = compare_metric("Pupil vs Right Edge",
                                left_eye["distances"]["pupil_to_right"], right_eye["distances"]["pupil_to_right"],
                                base_len, 0.08)

            angle_gaze_diff = abs(left_eye["angles"]["gaze_angle"] - right_eye["angles"]["gaze_angle"])
            decision_flag = not (m1["ok"] and m2["ok"] and m3["ok"])

            # Visualization - original images with markers
            try:
                # decode originals for plotting
                arrL = np.frombuffer(left_bytes, np.uint8)
                imgL = cv2.imdecode(arrL, cv2.IMREAD_COLOR)
                arrR = np.frombuffer(right_bytes, np.uint8)
                imgR = cv2.imdecode(arrR, cv2.IMREAD_COLOR)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                for ax, img, eye_data, name in zip(axes,
                                                   [imgL, imgR],
                                                   [left_eye, right_eye],
                                                   ["Left", "Right"]):
                    if img is None:
                        ax.text(0.5, 0.5, "Image decode failed", ha="center")
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    px = eye_data["pupil_center"]
                    sc = eye_data["sclera_center"]
                    le = eye_data["left_edge"]
                    re = eye_data["right_edge"]
                    # plot only if centers are non-zero
                    if px is not None:
                        ax.plot(px[0], px[1], 'ro', label='Pupil')
                    if sc is not None:
                        ax.plot(sc[0], sc[1], 'go', label='Sclera Center')
                    # edges may be tuples of ints
                    if isinstance(le, tuple):
                        ax.plot(le[0], le[1], 'bo', label='Left Edge')
                    if isinstance(re, tuple):
                        ax.plot(re[0], re[1], 'yo', label='Right Edge')
                    ax.set_title(f"{name} Eye")
                    ax.axis("off")
                axes[1].legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not render overlay plots: {e}")

            # show metrics
            with st.expander("📊 Detailed metrics and decision", expanded=True):
                st.markdown("**Distance metrics**")
                for m in (m1, m2, m3):
                    status = "OK" if m["ok"] else "✖ Possible issue"
                    st.write(f"- **{m['label']}**: Left={m['left']:.2f} | Right={m['right']:.2f} | Diff={m['diff']:.2f} | Threshold={m['threshold']:.2f} → **{status}**")
                st.markdown("**Angle metrics**")
                st.write(f"- Gaze angle diff = {angle_gaze_diff:.2f} degrees")
                st.markdown("**Decision**")
                if decision_flag:
                    st.error("⚠️ Possible Strabismus Detected!")
                else:
                    st.success("✅ Eye Alignment Looks Normal.")

                # small downloadable text report
                report_lines = []
                report_lines.append("Eye Strabismus Analyzer Report\n")
                for m in (m1, m2, m3):
                    report_lines.append(f"{m['label']}: Left={m['left']:.2f}, Right={m['right']:.2f}, Diff={m['diff']:.2f}, Threshold={m['threshold']:.2f}, OK={m['ok']}\n")
                report_lines.append(f"Gaze angle diff: {angle_gaze_diff:.2f}\n")
                report_lines.append("Decision: " + ("Possible Strabismus Detected" if decision_flag else "Eye Alignment Looks Normal") + "\n")
                report_bytes = ("\n".join(report_lines)).encode("utf-8")
                st.download_button("Download report (.txt)", report_bytes, file_name="eye_report.txt")

            # cleanup
            del modelS, modelP
            tf.keras.backend.clear_session()
            gc.collect()
