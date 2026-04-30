"""
Streamlit Application for Vehicle Type Classification
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import json
import os
from loguru import logger

# Import project modules
from src.data.data_loader import ConfigLoader
from src.data.data_preprocessing import DataPreprocessor
from src.models.predict import VehiclePredictor
from src.models.evaluate import ModelEvaluator


# Configure page
st.set_page_config(
    page_title="Vehicle Type Classification",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .confidence-high { color: #4CAF50; }
    .confidence-medium { color: #FF9800; }
    .confidence-low { color: #F44336; }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .footer {
        text-align: center;
        color: #9E9E9E;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)


class VehicleClassificationApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = self._load_config()
        self.predictor = None
        self.class_names = ['car', 'bike', 'truck', 'bus']
        self.class_icons = {
            'car': '🚗',
            'bike': '🏍️',
            'truck': '🚛',
            'bus': '🚌'
        }
        self.class_colors = {
            'car': '#1E88E5',
            'bike': '#43A047',
            'truck': '#FB8C00',
            'bus': '#E53935'
        }
    
    def _load_config(self):
        """Load application configuration"""
        try:
            return ConfigLoader()
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return None
    
    @st.cache_resource
    def load_predictor(_self):
        """Load the prediction model (cached)"""
        model_path = "models/best_model.pth"
        
        if not Path(model_path).exists():
            return None
        
        try:
            predictor = VehiclePredictor(model_path)
            predictor.warmup()
            return predictor
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.markdown('<p class="main-header">🚗 Vehicle Type Classification</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Upload an image to classify the vehicle type</p>', unsafe_allow_html=True)
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
        
        # Footer
        st.markdown('<p class="footer">Built with Streamlit & PyTorch | Vehicle Classification System</p>', unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar content"""
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Model information
            st.subheader("📊 Model Info")
            
            predictor = self.load_predictor()
            if predictor is not None:
                model_info = predictor.get_model_info()
                st.success("✅ Model loaded successfully")
                
                with st.expander("Model Details", expanded=False):
                    st.write(f"**Architecture:** {model_info.get('model_name', 'N/A')}")
                    st.write(f"**Classes:** {model_info.get('num_classes', 'N/A')}")
                    st.write(f"**Parameters:** {model_info.get('num_parameters', 'N/A'):,}")
                    st.write(f"**Device:** {model_info.get('device', 'N/A')}")
            else:
                st.warning("⚠️ Model not loaded")
                st.info("Please train a model first using the training script.")
            
            st.divider()
            
            # Classification categories
            st.subheader("🏷️ Vehicle Categories")
            for class_name in self.class_names:
                icon = self.class_icons.get(class_name, '🚙')
                color = self.class_colors.get(class_name, '#666')
                st.markdown(f"{icon} **{class_name.capitalize()}**")
            
            st.divider()
            
            # Confidence threshold
            st.subheader("🎯 Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence for a valid prediction"
            )
            
            show_probabilities = st.checkbox(
                "Show all probabilities",
                value=True,
                help="Display probabilities for all classes"
            )
            
            st.session_state['confidence_threshold'] = confidence_threshold
            st.session_state['show_probabilities'] = show_probabilities
            
            st.divider()
            
            # About section
            with st.expander("ℹ️ About"):
                st.markdown("""
                This application uses a Convolutional Neural Network (CNN) 
                with transfer learning to classify vehicle types.
                
                **Supported vehicles:**
                - 🚗 Cars
                - 🏍️ Bikes/Motorcycles
                - 🚛 Trucks
                - 🚌 Buses
                
                **Technology Stack:**
                - PyTorch for deep learning
                - EfficientNet for feature extraction
                - Streamlit for the web interface
                """)
    
    def _render_main_content(self):
        """Render main content area"""
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📸 Upload & Classify", "📊 Batch Processing", "📈 Model Evaluation"])
        
        with tab1:
            self._render_single_prediction()
        
        with tab2:
            self._render_batch_processing()
        
        with tab3:
            self._render_evaluation()
    
    def _render_single_prediction(self):
        """Render single image prediction interface"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a vehicle image for classification"
            )
            
            # Sample images
            st.markdown("---")
            st.markdown("**Or try a sample image:**")
            
            sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
            
            with sample_col1:
                if st.button("🚗 Car", use_container_width=True):
                    st.session_state['sample_class'] = 'car'
            with sample_col2:
                if st.button("🏍️ Bike", use_container_width=True):
                    st.session_state['sample_class'] = 'bike'
            with sample_col3:
                if st.button("🚛 Truck", use_container_width=True):
                    st.session_state['sample_class'] = 'truck'
            with sample_col4:
                if st.button("🚌 Bus", use_container_width=True):
                    st.session_state['sample_class'] = 'bus'
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            if uploaded_file is not None:
                self._process_and_display(uploaded_file)
            elif 'sample_class' in st.session_state:
                # Generate and display sample prediction
                self._display_sample_prediction(st.session_state['sample_class'])
            else:
                st.info("👆 Upload an image or select a sample to get started")
    
    def _process_and_display(self, uploaded_file):
        """Process uploaded image and display results"""
        # Load predictor
        predictor = self.load_predictor()
        
        if predictor is None:
            st.error("❌ Model not available. Please train a model first.")
            return
        
        # Read and display image
        image_bytes = uploaded_file.read()
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert to RGB if necessary
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Display original image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction
        with st.spinner("🔄 Analyzing image..."):
            start_time = time.time()
            
            try:
                result = predictor.predict(image_np, return_probs=True)
                inference_time = time.time() - start_time
                
                self._display_prediction_results(result, inference_time)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
    
    def _display_prediction_results(self, result, inference_time):
        """Display prediction results"""
        predicted_class = result['class']
        confidence = result['confidence']
        probabilities = result.get('probabilities', {})
        
        # Get confidence color
        if confidence >= 0.8:
            conf_class = "confidence-high"
            conf_emoji = "✅"
        elif confidence >= 0.5:
            conf_class = "confidence-medium"
            conf_emoji = "⚠️"
        else:
            conf_class = "confidence-low"
            conf_emoji = "❓"
        
        # Display main prediction
        icon = self.class_icons.get(predicted_class, '🚙')
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="text-align: center;">{icon} {predicted_class.upper()}</h2>
            <p style="text-align: center; font-size: 1.5rem;" class="{conf_class}">
                {conf_emoji} Confidence: {confidence*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Class", predicted_class.capitalize())
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Inference Time", f"{inference_time*1000:.1f}ms")
        
        # Display probability chart
        if st.session_state.get('show_probabilities', True):
            st.markdown("### 📊 Class Probabilities")
            
            # Create probability chart
            prob_df = {
                'Class': [c.capitalize() for c in probabilities.keys()],
                'Probability': [p * 100 for p in probabilities.values()]
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prob_df['Class'],
                    y=prob_df['Probability'],
                    marker_color=[self.class_colors.get(c, '#666') for c in probabilities.keys()],
                    text=[f"{p:.1f}%" for p in prob_df['Probability']],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Vehicle Class",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence threshold warning
        threshold = st.session_state.get('confidence_threshold', 0.5)
        if confidence < threshold:
            st.warning(f"⚠️ Confidence is below threshold ({threshold*100:.0f}%). Consider using a clearer image.")
    
    def _display_sample_prediction(self, sample_class):
        """Display sample prediction for demo purposes"""
        # Generate random sample prediction
        sample_probs = {
            'car': 0.1,
            'bike': 0.1,
            'truck': 0.1,
            'bus': 0.1
        }
        sample_probs[sample_class] = 0.85
        
        # Normalize
        total = sum(sample_probs.values())
        sample_probs = {k: v/total for k, v in sample_probs.items()}
        
        result = {
            'class': sample_class,
            'confidence': sample_probs[sample_class],
            'probabilities': sample_probs
        }
        
        st.info(f"📌 Showing sample prediction for: **{sample_class.capitalize()}**")
        self._display_prediction_results(result, 0.05)
    
    def _render_batch_processing(self):
        """Render batch processing interface"""
        st.subheader("📁 Batch Processing")
        st.markdown("Upload multiple images for batch classification")
        
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        
        if uploaded_files:
            predictor = self.load_predictor()
            
            if predictor is None:
                st.error("❌ Model not available")
                return
            
            if st.button("🚀 Process All Images", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    try:
                        image = Image.open(file)
                        image_np = np.array(image)
                        
                        if len(image_np.shape) == 2:
                            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                        
                        result = predictor.predict(image_np)
                        result['filename'] = file.name
                        results.append(result)
                        
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Display results
                st.markdown("### 📊 Batch Results")
                
                # Summary statistics
                successful = [r for r in results if 'error' not in r]
                class_counts = {}
                for r in successful:
                    c = r['class']
                    class_counts[c] = class_counts.get(c, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Successful", len(successful))
                with col3:
                    st.metric("Failed", len(results) - len(successful))
                with col4:
                    avg_conf = np.mean([r['confidence'] for r in successful]) if successful else 0
                    st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                
                # Class distribution chart
                if class_counts:
                    fig = px.pie(
                        names=list(class_counts.keys()),
                        values=list(class_counts.values()),
                        title="Vehicle Type Distribution",
                        color_discrete_map=self.class_colors
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("### 📋 Detailed Results")
                
                for result in results:
                    if 'error' in result:
                        st.error(f"❌ {result['filename']}: {result['error']}")
                    else:
                        icon = self.class_icons.get(result['class'], '🚙')
                        st.success(
                            f"{icon} **{result['filename']}**: {result['class'].capitalize()} "
                            f"({result['confidence']*100:.1f}%)"
                        )
    
    def _render_evaluation(self):
        """Render model evaluation interface"""
        st.subheader("📈 Model Evaluation")
        
        # Check for evaluation results
        results_path = "models/evaluation_results.json"
        history_path = "models/training_history.json"
        
        col1, col2 = st.columns(2)
        
        with col1:
            if Path(results_path).exists():
                with open(results_path, 'r') as f:
                    eval_results = json.load(f)
                
                st.markdown("### 📊 Evaluation Metrics")
                
                metrics = eval_results.get('metrics', {})
                
                # Overall metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
                with metric_col2:
                    st.metric("Precision", f"{metrics.get('precision_macro', 0)*100:.2f}%")
                with metric_col3:
                    st.metric("Recall", f"{metrics.get('recall_macro', 0)*100:.2f}%")
                with metric_col4:
                    st.metric("F1 Score", f"{metrics.get('f1_macro', 0)*100:.2f}%")
                
                # Per-class metrics
                st.markdown("### 🏷️ Per-Class Performance")
                
                class_names = eval_results.get('class_names', self.class_names)
                precision = metrics.get('precision_per_class', [])
                recall = metrics.get('recall_per_class', [])
                f1 = metrics.get('f1_per_class', [])
                
                if precision and recall and f1:
                    fig = go.Figure(data=[
                        go.Bar(name='Precision', x=class_names, y=[p*100 for p in precision]),
                        go.Bar(name='Recall', x=class_names, y=[r*100 for r in recall]),
                        go.Bar(name='F1 Score', x=class_names, y=[f*100 for f in f1])
                    ])
                    
                    fig.update_layout(
                        barmode='group',
                        title="Per-Class Metrics",
                        yaxis_title="Score (%)",
                        yaxis_range=[0, 100]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion matrix
                if 'confusion_matrix' in metrics:
                    st.markdown("### 🔢 Confusion Matrix")
                    
                    cm = np.array(metrics['confusion_matrix'])
                    
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="True", color="Count"),
                        x=class_names,
                        y=class_names,
                        text_auto=True,
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No evaluation results found. Run model evaluation first.")
        
        with col2:
            if Path(history_path).exists():
                with open(history_path, 'r') as f:
                    history = json.load(f)
                
                st.markdown("### 📈 Training History")
                
                # Loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history.get('train_loss', []),
                    mode='lines',
                    name='Train Loss'
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history.get('val_loss', []),
                    mode='lines',
                    name='Val Loss'
                ))
                fig_loss.update_layout(
                    title="Loss Over Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Loss"
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
                # Accuracy plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history.get('train_acc', []),
                    mode='lines',
                    name='Train Accuracy'
                ))
                fig_acc.add_trace(go.Scatter(
                    y=history.get('val_acc', []),
                    mode='lines',
                    name='Val Accuracy'
                ))
                fig_acc.update_layout(
                    title="Accuracy Over Epochs",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy (%)"
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                st.info("📈 No training history found. Train a model first.")


def main():
    """Main entry point"""
    app = VehicleClassificationApp()
    app.run()


if __name__ == "__main__":
    main()