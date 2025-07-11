import streamlit as st
import cv2
import numpy as np
import pickle
import mahotas
from skimage.feature import local_binary_pattern

# Load trained model
with open("C:/Users/dell/OneDrive/Desktop/pandas demo/New folder/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("C:/Users/dell/OneDrive/Desktop/pandas demo/New folder/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Feature extraction function
def extract_features(image):
    try:
        image = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0,10), range=(0,9))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        haralick = mahotas.features.haralick(gray).mean(axis=0)

        return np.hstack([hist, lbp_hist, haralick])
    except:
        return None

# Streamlit UI
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title(" Plant Disease Classifier (SVM + Handcrafted Features)")
st.markdown("Upload a plant leaf image. The app will predict the disease using handcrafted image features and a trained SVM model.")

uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Extracting features and making prediction..."):
        features = extract_features(image)
        if features is not None:
            pred = model.predict([features])[0]
            pred_label = le.inverse_transform([pred])[0]
            st.success(f" Predicted Disease: **{pred_label.upper()}**")
        else:
            st.error(" Could not extract features from the image.")
