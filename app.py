import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import json
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model and tokenizer with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to your extracted model folder
    MODEL_PATH = "./bert-news-classifier"  # Update this path
    
    try:
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        
        with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
            mapping = json.load(f)
            id2label = mapping["id2label"]
            label2id = mapping["label2id"]
        
        return model, tokenizer, id2label, label2id, device, True
    
    except Exception as e:
        return None, None, None, None, None, False

def predict(text, model, tokenizer, id2label, device):
    """Make prediction on input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()
        all_probs = probs[0].cpu().numpy()
    
    return predicted_class, confidence, all_probs

def create_probability_chart(all_probs, id2label, predicted_class):
    """Create animated probability chart"""
    labels = [id2label[str(i)] for i in range(len(all_probs))]
    colors = ['#ff6b6b' if i == predicted_class else '#e0e0e0' 
              for i in range(len(all_probs))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=all_probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in all_probs],
            textposition='auto',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Prediction Confidence',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        xaxis_title='Category',
        showlegend=False,
        height=400,
        template='plotly_white',
        transition_duration=500
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    model, tokenizer, id2label, label2id, device, loaded = load_model()
    
    if loaded:
        st.success("✅ Model Loaded")
        st.info(f"Device: {device}")
        st.info(f"Model: DistilBERT")
        st.info(f"Classes: {len(id2label)}")
        
        st.markdown("---")
        st.header("📈 Model Performance")
        st.metric("Accuracy", "90.5%")
        st.metric("F1 Score", "90.5%")
        
        st.markdown("---")
        st.header("🏷️ Categories")
        for i, label in id2label.items():
            st.write(f"**{i}**: {label}")
    else:
        st.error("❌ Model not found")
        st.info("Please ensure model files are in the correct directory")

# Main content
st.markdown('<p class="main-header">📰 News Topic Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by DistilBERT | Classify news into World, Sports, Business, or Sci/Tech</p>', unsafe_allow_html=True)

# Input section
st.header("📝 Enter News Headline")

col1, col2 = st.columns([3, 1])

with col1:
    text_input = st.text_area(
        "News text:",
        height=120,
        placeholder="e.g., Apple unveils revolutionary AI features in latest iPhone release...",
        label_visibility="collapsed"
    )

with col2:
    st.write("")  # Spacer
    st.write("")
    predict_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Example buttons
    st.markdown("**Quick Examples:**")
    if st.button("Tech News", use_container_width=True):
        text_input = "Apple announces new AI chip for iPhone 15"
        st.session_state['text'] = text_input
    if st.button("Sports", use_container_width=True):
        text_input = "Manchester United wins Premier League championship"
        st.session_state['text'] = text_input

if clear_btn:
    st.session_state.clear()
    st.rerun()

# Prediction
if predict_btn and text_input and loaded:
    with st.spinner("🤖 Analyzing..."):
        predicted_class, confidence, all_probs = predict(
            text_input, model, tokenizer, id2label, device
        )
        
        predicted_label = id2label[str(predicted_class)]
        
        # Results section
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Predicted Category",
                value=predicted_label,
                delta=f"{confidence:.1%} confidence"
            )
        
        with col2:
            # Category icon
            icons = {
                "World": "🌍",
                "Sports": "⚽",
                "Business": "💼",
                "Sci/Tech": "🔬"
            }
            st.metric(
                label="Category Icon",
                value=icons.get(predicted_label, "📰")
            )
        
        with col3:
            # Confidence level
            if confidence > 0.9:
                level = "High"
                color = "green"
            elif confidence > 0.7:
                level = "Medium"
                color = "orange"
            else:
                level = "Low"
                color = "red"
            
            st.metric(
                label="Confidence Level",
                value=level
            )
        
        # Probability chart
        st.plotly_chart(
            create_probability_chart(all_probs, id2label, predicted_class),
            use_container_width=True
        )
        
        # Interpretation
        st.info(f"""
        **Interpretation:** This news is classified as **{predicted_label}** with 
        **{confidence:.1%}** confidence. The model is 
        {'very confident' if confidence > 0.9 else 'moderately confident' if confidence > 0.7 else 'uncertain'} 
        about this prediction.
        """)
        
        # All probabilities table
        with st.expander("📊 View All Probabilities"):
            prob_df = pd.DataFrame({
                'Category': [id2label[str(i)] for i in range(len(all_probs))],
                'Probability': [f"{p:.2%}" for p in all_probs],
                'Confidence Score': [f"{p:.4f}" for p in all_probs]
            })
            prob_df = prob_df.sort_values('Confidence Score', ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

elif predict_btn and not text_input:
    st.warning("⚠️ Please enter some text to classify.")

elif predict_btn and not loaded:
    st.error("❌ Model not loaded. Please check the model path.")

# Footer
st.markdown("---")
st.caption("""
    **DevelopersHub Corporation** | AI/ML Engineering Internship Task 1  
    Model: DistilBERT | Accuracy: 90.5% | Dataset: AG News
""")