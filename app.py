import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import cv2
import os

#hah
def generate_gradcam(img_array, model):
    try:
        base_model = model.layers[0]
        classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
        x = classifier_input
        for layer in model.layers[1:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_outputs = base_model(img_array)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None: return None

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-10)
        return heatmap
    except Exception as e:
        return None

# --- Configuration & Resources ---
st.set_page_config(page_title="GeoVision Pro | LULC Analyzer", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_resources():
    model_path = 'models/your_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    _ = model(tf.zeros((1, 128, 128, 3)))
    return model

model = load_resources()
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=60)
    st.title("LULC Diagnostics")
    st.metric("Overall Accuracy", "95.02%", help="Evaluated on validation dataset")
    st.metric("Kappa Coefficient", "0.91", help="Statistical measure of inter-rater reliability")
    st.divider()
    st.success("XAI Engine: Graph-Connected Grad-CAM")
    st.info("Normalization: Raw Pixel Array (0-255)")

#ui
st.title("🛰️ Land Use & Cover Classification")
st.markdown("Upload satellite imagery to analyze land cover utilizing CNN inference and Explainable AI.")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Satellite Feed", type=["jpg", "png", "jpeg"], help="Supported formats: JPG, PNG")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    
    # EXACT SAME PREPROCESSING (Untouched)
    img_resized = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_final = np.expand_dims(img_array, axis=0) 

    # EXACT SAME INFERENCE (Untouched)
    with st.spinner('Running Neural Inference & Generating Gradients...'):
        preds = model.predict(img_final)
        probs = tf.nn.softmax(preds[0]).numpy()
        idx = np.argmax(probs)

    # UI LAYOUT
    left, right = st.columns([1, 1.2], gap="large")
    
    with left:
        st.image(image, caption="Original Satellite Feed", use_container_width=True)
        
        # NEW: The Advanced Diagnostics Expander!
        with st.expander("🔬 Advanced Diagnostics: Full Class Breakdown"):
            st.caption("Raw Softmax Probability Distribution")
            # Sort the probabilities from highest to lowest for display
            sorted_indices = np.argsort(probs)[::-1]
            for i in sorted_indices:
                class_name = class_names[i]
                confidence = probs[i] * 100
                # Use st.progress for visual bars and bold text for the winner
                if i == idx:
                    st.markdown(f"**{class_name}: {confidence:.2f}%** ")
                    st.progress(int(confidence) if int(confidence) > 0 else 1)
                else:
                    st.markdown(f"{class_name}: {confidence:.2f}%")
                    # Progress bar requires an int between 0 and 100
                    bar_val = int(confidence)
                    if bar_val > 0: st.progress(bar_val)

    with right:
        # Sleeker Prediction Card
        st.markdown(f"""
            <div style="padding:20px; border-radius:15px; background:linear-gradient(145deg, #1e293b, #0f172a); border:1px solid #3b82f6; text-align:center; margin-bottom: 20px;">
                <h4 style="color:#94a3b8; margin:0; text-transform: uppercase; letter-spacing: 1px;">Primary Classification</h4>
                <h1 style="color:#60a5fa; margin:10px 0; font-size: 2.5rem;">{class_names[idx]}</h1>
                <h3 style="color:#e2e8f0; margin:0;">Confidence: {probs[idx]*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Probability Chart", "🔄 Preprocessing Preview", "🔍 Neural Focus Map"])
        
        with tab1:
            fig = px.bar(x=probs, y=class_names, orientation='h', color=probs, 
                         color_continuous_scale='Blues', title="Class Probability Distribution")
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("#### Spatial Invariance (Augmentation Preview)")
            st.caption("How the model views the data during training to prevent overfitting.")
            img_np = np.array(image)
            c1, c2 = st.columns(2)
            c1.image(cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE), caption="90° Rotation")
            c2.image(cv2.flip(img_np, 1), caption="Horizontal Flip")

        with tab3:
            st.write("#### Grad-CAM Explainable AI")
            st.caption("Heatmap highlighting the specific features driving the model's decision.")
            
            heatmap = generate_gradcam(img_final, model)
            if heatmap is not None:
                h_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
                h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
                h_colored = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)
                superimposed = cv2.addWeighted(np.array(image), 0.6, h_colored, 0.4, 0)
                st.image(superimposed, use_container_width=True, caption="Red = High Influence | Blue = Ignored by CNN")
            else:
                st.error("Gradient Error: Model is saturated for this specific image.")
else:
    st.info("👈 Please upload a satellite image from the sidebar or drag and drop one here to begin the analysis.")