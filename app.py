import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="MindGuard", page_icon="", layout="wide")

# --- 2. THE COMPACT AESTHETIC CSS ---
st.markdown("""
    <style>
    .stApp { background: #0f172a; color: #f8fafc; }
    
    /* Header Styling */
    .header { font-size: 32px; font-weight: 800; background: linear-gradient(90deg, #3b82f6, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }

    /* Compact Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }

    /* Result Typography */
    .res-label { font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .res-value { font-size: 36px; font-weight: 900; margin: 5px 0; }
    
    /* Fix buttons */
    .stButton>button { background: #3b82f6; color: white; border-radius: 8px; font-weight: bold; width: 100%; border: none; height: 45px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC ---
@st.cache_resource
def load_assets():
    checkpoint = torch.load('mental_health_final.pth', map_location=torch.device('cpu'), weights_only=False)
    classes = checkpoint['classes']
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(classes))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer, classes

def analyze_text(text, model, tokenizer, classes):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
    return classes[idx.item()], conf.item() * 100

# --- 4. DASHBOARD LAYOUT ---
st.markdown('<div class="header">MindGuard</div>', unsafe_allow_html=True)

col_input, col_output = st.columns([1.2, 1])

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📝 Text Input", "📁 PDF Upload"])
    
    input_text = ""
    with tab1:
        input_text = st.text_area("Analyze Social Media Content", placeholder="Enter text here...", height=200)
    
    with tab2:
        uploaded_file = st.file_uploader("Upload a report or post (PDF)", type="pdf")
        if uploaded_file:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                input_text += page.extract_text()
            st.success("PDF Content Extracted!")

    analyze_btn = st.button("RUN NEURAL DIAGNOSIS")
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    if analyze_btn and input_text.strip():
        with st.spinner("Processing..."):
            model, tokenizer, classes = load_assets()
            prediction, confidence = analyze_text(input_text, model, tokenizer, classes)
            
            # Dynamic Color Selection
            color = "#10b981" if prediction == "Normal" else "#ef4444"
            if prediction in ["Stress", "Anxiety"]: color = "#f59e0b"

            st.markdown(f"""
                <div class="glass-card">
                    <p class="res-label">Analysis Result</p>
                    <div class="res-value" style="color: {color};">{prediction}</div>
                    <p style="color: #94a3b8;">Confidence Level: <b>{confidence:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics Section
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Characters", len(input_text))
            m2.metric("Processing", "BERT-v1")
            st.markdown('</div>', unsafe_allow_html=True)

            if prediction in ["Suicidal", "Depression"] and confidence > 70:
                st.warning("⚠️ **Alert:** High priority patterns detected. Support resources advised.")
    else:
        # Placeholder when no analysis is run
        st.markdown("""
            <div class="glass-card" style="height: 350px; display: flex; align-items: center; justify-content: center;">
                <p style="color: #475569;">Analysis results will appear here after running diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #475569; font-size: 11px; margin-top: 50px;'>Final Year Project | Engineering Deployment v2.1</p>", unsafe_allow_html=True)