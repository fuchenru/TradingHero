import finnhub
from datetime import date, timedelta
import pandas as pd
import requests
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession, Part
import vertexai.preview.generative_models as generative_models
import re
vertexai.init(project="adsp-capstone-trading-hero", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-002")
finnhub_client = finnhub.Client(api_key="co6v709r01qj6a5mgco0co6v709r01qj6a5mgcog")
tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("fuchenru/Trading-Hero-LLM")
nlp = pipeline("text-classification", model=sentiment_model, tokenizer=tokenizer)

def preprocess(text, tokenizer, max_length=128):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    return inputs

def predict_sentiment(input_text):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=512
    )
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: 'Neutral 😐', 1: 'Positive 🙂', 2: 'Negative 😡'}
    predicted_sentiment = label_map[predicted_label]
    return predicted_sentiment

def get_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return pd.DataFrame()
        top_5_news = df.head(15)
        top_5_news['Sentiment Analysis'] = top_5_news['summary'].apply(predict_sentiment)
        return top_5_news
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

def get_all_stock_news(ticker_symbol, start_date, end_date):
    try:
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary'])
        if df.empty:
            return pd.Series()
        return df['headline']
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return pd.Series()

def extract_text_from_response(response):
    if response.candidates and response.candidates[0].content.parts:
        generated_text = response.candidates[0].content.parts[0].text.strip()
        return generated_text
    else:
        return "No generated content available."

def generate_vertexai_newsresponse(newsprompt, data):
    formatted_data = f"- News Summary: {data}"
    full_prompt = newsprompt + formatted_data
    responses = model.generate_content(full_prompt)
    return extract_text_from_response(responses)


def get_company_profile(ticker_symbol):
    """Get company profile information from Finnhub API"""
    try:
        profile = finnhub_client.company_profile2(symbol=ticker_symbol)
        return profile
    except Exception as e:
        st.error(f"Error fetching company profile: {e}")
        return None

def generate_company_keywords(ticker_symbol):
    """Dynamically generate company-specific keywords based on company profile"""
    profile = get_company_profile(ticker_symbol)
    if not profile:
        return {}
    
    # Extract basic company information
    company_name = profile.get('name', '')
    industry = profile.get('finnhubIndustry', '')
    
    # Split company name into parts for better matching
    name_parts = company_name.split()
    company_names = [company_name] + name_parts
    
    # Industry-specific keyword mappings
    industry_keywords = {
        'Technology': {
            'products': [
                'software', 'app', 'platform', 'cloud', 'AI', 'data', 'digital',
                'hardware', 'semiconductor', 'chip', 'processor', 'device', 'smartphone',
                'laptop', 'tablet', 'wearable', 'IoT', 'robot', 'automation', 'API',
                'blockchain', 'cryptocurrency', 'virtual reality', 'VR', 'AR', 'augmented reality',
                'network', '5G', '6G', 'quantum', 'machine learning', 'ML'
            ],
            'business_areas': [
                'technology', 'innovation', 'digital transformation', 
                'artificial intelligence', 'cloud computing', 'cybersecurity',
                'software development', 'IT services', 'data analytics',
                'semiconductor', 'telecommunications', 'internet services',
                'e-commerce', 'digital payments', 'enterprise software',
                'gaming', 'metaverse', 'quantum computing', 'edge computing',
                'DevOps', 'SaaS', 'PaaS', 'IaaS'
            ]
        },
        'Finance': {
            'products': [
                'banking', 'investment', 'loan', 'mortgage', 'trading', 'payment',
                'credit card', 'debit card', 'insurance', 'mutual fund', 'ETF',
                'cryptocurrency', 'bond', 'stock', 'option', 'future', 'derivative',
                'pension', 'retirement', 'savings', 'checking', 'wire transfer',
                'mobile payment', 'digital wallet', 'BNPL', 'factoring'
            ],
            'business_areas': [
                'financial services', 'banking', 'investment management', 
                'wealth management', 'fintech', 'insurance', 'real estate',
                'capital markets', 'asset management', 'private equity',
                'venture capital', 'commercial banking', 'retail banking',
                'investment banking', 'risk management', 'compliance',
                'treasury', 'credit services', 'payment processing'
            ]
        },
        'Healthcare': {
            'products': [
                'drug', 'treatment', 'therapy', 'device', 'vaccine',
                'diagnostic', 'implant', 'pharmaceutical', 'medicine',
                'medical equipment', 'surgical tool', 'imaging system',
                'telehealth', 'wearable', 'biotechnology', 'genomic',
                'prosthetic', 'orthopedic', 'dental', 'ophthalmic',
                'therapeutic', 'antibody', 'protein', 'cell therapy'
            ],
            'business_areas': [
                'healthcare', 'pharmaceutical', 'biotech', 'medical devices', 
                'clinical trials', 'research', 'drug development',
                'patient care', 'diagnostics', 'hospital management',
                'healthcare IT', 'telemedicine', 'precision medicine',
                'genomics', 'life sciences', 'regulatory compliance',
                'clinical research', 'medical research', 'pharmacy'
            ]
        },
        'Retail': {
            'products': [
                'merchandise', 'consumer goods', 'apparel', 'electronics',
                'groceries', 'furniture', 'appliances', 'home goods',
                'beauty products', 'fashion', 'luxury goods', 'accessories',
                'sporting goods', 'toys', 'books', 'automotive parts',
                'home improvement', 'pet supplies', 'office supplies'
            ],
            'business_areas': [
                'retail', 'e-commerce', 'brick and mortar', 'online retail',
                'wholesale', 'distribution', 'supply chain', 'inventory management',
                'merchandising', 'customer service', 'loyalty program',
                'point of sale', 'omnichannel', 'direct-to-consumer',
                'retail analytics', 'store operations', 'digital retail'
            ]
        },
        'Energy': {
            'products': [
                'oil', 'gas', 'solar panel', 'wind turbine', 'battery',
                'nuclear', 'hydroelectric', 'biofuel', 'renewable energy',
                'electric vehicle', 'charging station', 'smart grid',
                'energy storage', 'power plant', 'carbon capture',
                'hydrogen fuel', 'geothermal', 'biomass', 'tidal energy'
            ],
            'business_areas': [
                'energy production', 'renewable energy', 'oil and gas',
                'utilities', 'power generation', 'energy distribution',
                'clean energy', 'sustainability', 'energy storage',
                'grid infrastructure', 'energy trading', 'energy efficiency',
                'carbon reduction', 'environmental services', 'green energy',
                'energy management', 'energy technology'
            ]
        },
        'Manufacturing': {
            'products': [
                'machinery', 'equipment', 'industrial goods', 'automotive',
                'aerospace', 'electronics', 'components', 'materials',
                'chemicals', 'plastics', 'metals', 'textiles', 'packaging',
                'construction materials', 'industrial supplies', 'tools',
                'robotics', 'automation equipment', 'sensors'
            ],
            'business_areas': [
                'manufacturing', 'production', 'assembly', 'quality control',
                'supply chain', 'logistics', 'industrial automation',
                'process improvement', 'inventory management', 'procurement',
                'industry 4.0', 'smart manufacturing', 'lean manufacturing',
                'factory automation', 'production planning', 'maintenance'
            ]
        },
        'Transportation': {
            'products': [
                'vehicle', 'aircraft', 'ship', 'train', 'truck', 'bus',
                'electric vehicle', 'autonomous vehicle', 'drone', 'bicycle',
                'scooter', 'locomotive', 'cargo container', 'fleet management',
                'navigation system', 'logistics software', 'tracking system'
            ],
            'business_areas': [
                'transportation', 'logistics', 'shipping', 'aviation',
                'railways', 'maritime', 'freight', 'delivery services',
                'fleet management', 'public transit', 'ride-sharing',
                'autonomous vehicles', 'urban mobility', 'last-mile delivery',
                'supply chain logistics', 'transportation infrastructure'
            ]
        },
        'Entertainment': {
            'products': [
                'movie', 'show', 'game', 'music', 'streaming service',
                'content', 'media', 'video', 'podcast', 'live event',
                'broadcast', 'film', 'television', 'digital content',
                'animation', 'virtual event', 'interactive media'
            ],
            'business_areas': [
                'media', 'entertainment', 'streaming', 'broadcasting',
                'gaming', 'film production', 'music industry', 'publishing',
                'digital media', 'content creation', 'live entertainment',
                'sports entertainment', 'interactive entertainment',
                'digital distribution', 'content licensing'
            ]
        },
        'Real Estate': {
            'products': [
                'property', 'building', 'development', 'construction',
                'commercial property', 'residential property', 'land',
                'apartment', 'office space', 'retail space', 'warehouse',
                'industrial space', 'hotel', 'resort', 'housing'
            ],
            'business_areas': [
                'real estate development', 'property management',
                'construction', 'commercial real estate', 'residential real estate',
                'property investment', 'facility management', 'leasing',
                'property technology', 'real estate services', 'architecture',
                'urban planning', 'property maintenance', 'real estate finance'
            ]
        }
    }
    
    # Get industry-specific keywords or use default ones
    industry_specific = industry_keywords.get(industry, {
        'products': ['product', 'service', 'solution'],
        'business_areas': ['business', 'market', 'industry']
    })
    
    # Combine everything into a comprehensive keyword set
    keywords = {
        'company_names': company_names,
        'products': industry_specific['products'],
        'business_areas': industry_specific['business_areas'] + [industry],
        'common_terms': [
            'announces', 'launches', 'reports', 'earnings',
            'revenue', 'growth', 'expansion', 'partnership',
            'acquisition', 'merger', 'investment'
        ]
    }
    
    return keywords

def get_filtered_stock_news(ticker_symbol, start_date, end_date):
    """
    Get and filter relevant company news with dynamic keyword generation
    """
    try:
        # Get raw news
        news = finnhub_client.company_news(ticker_symbol, _from=start_date, to=end_date)
        df = pd.DataFrame.from_records(news, columns=['headline', 'summary', 'datetime'])
        
        if df.empty:
            return pd.DataFrame()
        
        # Generate company-specific keywords
        company_keywords = generate_company_keywords(ticker_symbol)
        
        # Common irrelevant patterns
        irrelevant_patterns = [
            r'etf', r'index fund', r'market update', r'market close',
            r'stock market today', r'trading session', r'market wrap',
            r'stocks to watch', r'trading ideas', r'technical analysis'
        ]
        
        def calculate_relevance_score(row):
            """Calculate relevance score based on dynamic keyword matches"""
            text = f"{row['headline']} {row['summary']}".lower()
            
            # Check for irrelevant patterns (negative score)
            irrelevant_score = sum(1 for pattern in irrelevant_patterns 
                                 if re.search(pattern, text, re.IGNORECASE))
            
            # Calculate positive scores based on keyword matches
            keyword_scores = {
                'company_names': 3,  # Higher weight for company name matches
                'products': 2,       # Medium weight for product matches
                'business_areas': 2, # Medium weight for business area matches
                'common_terms': 1    # Lower weight for common business terms
            }
            
            total_score = 0
            for category, keywords in company_keywords.items():
                weight = keyword_scores.get(category, 1)
                matches = sum(1 for keyword in keywords 
                            if keyword.lower() in text)
                total_score += matches * weight
            
            return total_score - (irrelevant_score * 2)  # Higher penalty for irrelevant matches
        
        # Add relevance score and datetime conversion
        df = df[df['datetime'] > 0]
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
        df = df.dropna(subset=['datetime'])
        df['relevance_score'] = df.apply(calculate_relevance_score, axis=1)
        
        # Filter and sort
        filtered_df = df[df['relevance_score'] > 0].sort_values(
            by=['relevance_score', 'datetime'], 
            ascending=[False, False]
        )
        
        # Add sentiment analysis
        if not filtered_df.empty:
            filtered_df['Sentiment Analysis'] = filtered_df['headline'].apply(predict_sentiment)
            
            # Format datetime for display
            filtered_df['Date'] = filtered_df['datetime'].dt.strftime('%Y-%m-%d')
            
            # Return relevant columns
            return filtered_df.head(15)[['headline', 'Date', 'Sentiment Analysis']]
        
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()
