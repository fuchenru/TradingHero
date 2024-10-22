# Trading Hero Financial Sentiment Analysis

Model Description: This model is a fine-tuned version of [FinBERT](https://huggingface.co/yiyanghkust/finbert-pretrain), a BERT model pre-trained on financial texts. The fine-tuning process was conducted to adapt the model to specific financial NLP tasks, enhancing its performance on domain-specific applications for sentiment analysis.

See more detail on my Hugging Face page 🤗: https://huggingface.co/fuchenru/Trading-Hero-LLM 

## Model Use

Primary Users: Financial analysts, NLP researchers, and developers working on financial data.

## Training Data

Training Dataset: The model was fine-tuned on a custom dataset of financial communication texts. The dataset was split into training, validation, and test sets as follows:

Training Set: 10,918,272 tokens

Validation Set: 1,213,184 tokens

Test Set: 1,347,968 tokens

Pre-training Dataset: FinBERT was pre-trained on a large financial corpus totaling 4.9 billion tokens, including:

Corporate Reports (10-K & 10-Q): 2.5 billion tokens

Earnings Call Transcripts: 1.3 billion tokens

Analyst Reports: 1.1 billion tokens

## Evaluation

* Test Accuracy = 0.908469
* Test Precision = 0.927788
* Test Recall = 0.908469
* Test F1 = 0.913267
* **Labels**: 0 -> Neutral; 1 -> Positive; 2 -> Negative


## Usage

```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
model = AutoModelForSequenceClassification.from_pretrained("fuchenru/Trading-Hero-LLM")
nlp = pipeline("text-classification", model=model, tokenizer=tokenizer)
```

```
# Preprocess the input text
def preprocess(text, tokenizer, max_length=128):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return inputs

# Function to perform prediction
def predict_sentiment(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()

    # Map the predicted label to the original labels
    label_map = {0: 'neutral', 1: 'positive', 2: 'negative'}
    predicted_sentiment = label_map[predicted_label]

    return predicted_sentiment

stock_news = [
    "Market analysts predict a stable outlook for the coming weeks.",
    "The market remained relatively flat today, with minimal movement in stock prices.",
    "Investor sentiment improved following news of a potential trade deal.",
.......
]


for i in stock_news:
    predicted_sentiment = predict_sentiment(i)
    print("Predicted Sentiment:", predicted_sentiment)
```

```
Predicted Sentiment: neutral
Predicted Sentiment: neutral
Predicted Sentiment: positive
```

## Citation

```
@misc{yang2020finbert,
    title={FinBERT: A Pretrained Language Model for Financial Communications},
    author={Yi Yang and Mark Christopher Siy UY and Allen Huang},
    year={2020},
    eprint={2006.08097},
    archivePrefix={arXiv},
    }
```
