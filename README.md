# BERT

News Topic Classifier Using BERT

https://python.org
https://huggingface.co/transformers
https://pytorch.org
https://streamlit.io

Fine-tuned DistilBERT transformer model for classifying news headlines into 4 categories: World, Sports, Business, and Sci/Tech.

📊 Project Overview
Metric	Value
Base Model	distilbert-base-uncased
Fine-tuned On	AG News Dataset (50,000 samples)
Accuracy	90.5%
F1-Macro	90.5%
F1-Weighted	90.5%
Classes	4 (World, Sports, Business, Sci/Tech)

🎯 Features
✅ Transformer-based NLP (DistilBERT architecture)
✅ Transfer Learning (pre-trained → fine-tuned)
✅ Real-time Classification (Streamlit web app)
✅ Probability Visualization (interactive charts)
✅ Confidence Scoring (High/Medium/Low indicators)
✅ Fast Inference (~450 samples/sec on GPU)

🏗️ Project Structure
bert-news-classifier/
├── app.py                          # Streamlit web application
├── bert-news-classifier/           # Saved model folder
│   ├── config.json                 # Model configuration
│   ├── model.safetensors           # Model weights
│   ├── tokenizer.json              # Tokenizer vocabulary
│   ├── tokenizer_config.json       # Tokenizer config
│   └── label_mapping.json          # Class labels
├── requirements.txt                # Python dependencies
├── confusion_matrix.png            # Evaluation visualization
├── training_history.png            # Training curves
└── README.md       
              
🚀 Quick Start

1. Clone Repository
bash
git clone https://github.com/yourusername/bert-news-classifier.git
cd bert-news-classifier

2. Install Dependencies
bash
pip install -r requirements.txt

3. Run Streamlit App
bash
streamlit run app.py
4. Open Browser

Navigate to http://localhost:8501
📈 Model Performance
Classification Report

              precision    recall  f1-score   support

       World       0.90      0.94      0.92       242
      Sports       0.96      0.98      0.97       255
    Business       0.86      0.87      0.87       235
    Sci/Tech       0.89      0.84      0.86       268

    accuracy                           0.91      1000
   macro avg       0.90      0.91      0.90      1000
weighted avg       0.90      0.91      0.90      1000
Per-Class Performance

Class	Precision	Recall	F1-Score
Sports	96%	98%	97% ⭐
World	90%	94%	92%
Business	86%	87%	87%
Sci/Tech	89%	84%	86%

🔧 Model Architecture
Input Text
    ↓
[DistilBERT Tokenizer]
    ├── WordPiece tokenization
    ├── Add [CLS] and [SEP] tokens
    └── Pad/truncate to 64 tokens
    ↓
[DistilBERT Encoder]
    ├── 6 Transformer layers
    ├── Self-attention mechanism
    └── Hidden size: 768
    ↓
[Classification Head]
    ├── Dropout (0.1)
    ├── Linear (768 → 4)
    └── Softmax
    ↓
Output: Class probabilities

🧠 Key Techniques
Technique	Description
Transfer Learning	Started with pre-trained DistilBERT
Fine-tuning	Trained classification head + some layers
Dynamic Padding	Efficient batch processing
Mixed Precision (FP16)	2x faster training on GPU
Early Stopping	Prevent overfitting

🌐 Web Interface Features
Feature	Description
Text Input	Paste news headline or article
Real-time Prediction	Instant classification
Probability Chart	Interactive bar visualization
Confidence Metrics	Percentage + High/Medium/Low
Quick Examples	One-click sample texts
Detailed Breakdown	All class probabilities

📋 Tokenization Details
Parameter	Value	Reason
Max Length	64	News headlines are short
Tokenizer	WordPiece	Subword tokenization
Vocabulary	30,522 tokens	BERT vocabulary
Special Tokens	[CLS], [SEP], [PAD]	BERT standard
Example:
Input:  "Apple launches new iPhone"
Tokens: ["apple", "launches", "new", "iphone"]
IDs:    [101, 7128, 11834, 2047, 2570, 102]
        [CLS]  apple  launches  new  iphone  [SEP]

🛠️ Technologies Used
Category	Tools
Deep Learning	PyTorch, Transformers
NLP Model	DistilBERT (Hugging Face)
Web Framework	Streamlit
Visualization	Plotly
Deployment	joblib, safetensors

📊 Dataset
Source: AG News (Hugging Face Datasets)
Total Samples: 120,000 (used 50,000 for training)
Classes: 4 balanced categories
Text: Title + Description combined
Split: 50k train / 1k val / 1k test

Class Distribution
Class	Train	Test
World	12,500	250
Sports	12,500	250
Business	12,500	250
Sci/Tech	12,500	250

🎯 Training Configuration
Parameter	Value
Epochs	2
Batch Size	64
Learning Rate	3e-5
Optimizer	AdamW
Weight Decay	0.01
FP16	Enabled
Max Sequence Length	64

📝 Sample Predictions
Input	Prediction	Confidence
"Apple unveils new AI chip for iPhone"	Sci/Tech	94.5%
"Manchester United wins Premier League"	Sports	97.2%
"Federal Reserve announces interest rate hike"	Business	89.3%
"NASA discovers water on Mars"	Sci/Tech	91.8%

🚀 Inference Speed
Device	Speed
GPU (Tesla T4)	~450 samples/sec
CPU	~50 samples/sec
🔮 Future Improvements
[ ] Try BERT-base for higher accuracy
[ ] Multi-label classification
[ ] Deploy on Hugging Face Spaces
[ ] Add explainability (attention visualization)
[ ] Support longer articles (>512 tokens)

📄 Model Files
File	Size	Description
model.safetensors	~250 MB	Model weights
tokenizer.json	~1 MB	Tokenization vocabulary
config.json	~1 KB	Architecture config
label_mapping.json	~100 B	Class label mapping

💾 Loading the Model
Python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

# Load model and tokenizer
model_path = "./bert-news-classifier"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load label mapping
with open(f"{model_path}/label_mapping.json") as f:
    labels = json.load(f)["id2label"]

# Predict
text = "Your news headline here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()

print(f"Prediction: {labels[str(prediction)]} ({confidence:.2%})")
