import requests
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize

# Download required NLTK packages
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Step 1: Fetch a news article using NewsAPI
API_KEY = '755a8c08f4dc4ca9bf064f122c0241de'
NEWS_API_URL = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'

params = {
    'apiKey': API_KEY,
    'country': 'us',
    'category': 'technology',  # You can choose any category
    'pageSize': 1
}

response = requests.get(NEWS_API_URL, params=params)
data = response.json()

# Extract the content of the first article
article = data['articles'][0]
title = article['title']
description = article['description']
content = article['content']

news_article = f"{title}\n\n{description}\n\n{content}"
print("News Article:\n", news_article)

# Step 2: Extract named entities using SpaCy
# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Process the news article
doc = nlp(news_article)

# Extract named entities
spacy_entities = [(entity.text, entity.label_) for entity in doc.ents]
print("\nSpaCy Named Entities:")
for entity in spacy_entities:
    print(entity)

# Step 3: Extract named entities using NLTK
# Tokenize the article into sentences
sentences = sent_tokenize(news_article)

# Tokenize, POS tagging, and Named Entity Chunking
nltk_entities = []
for sentence in sentences:
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    entities = ne_chunk(tagged)
    nltk_entities.extend([chunk for chunk in entities if hasattr(chunk, 'label')])

print("\nNLTK Named Entities:")
for entity in nltk_entities:
    print(' '.join(c[0] for c in entity), entity.label())

# Step 4: Compare the results
spacy_entity_set = set(spacy_entities)
nltk_entity_set = set((' '.join(c[0] for c in entity), entity.label()) for entity in nltk_entities)

print("\nComparison of SpaCy and NLTK Named Entities:")
print("\nEntities in SpaCy but not in NLTK:")
for entity in spacy_entity_set - nltk_entity_set:
    print(entity)

print("\nEntities in NLTK but not in SpaCy:")
for entity in nltk_entity_set - spacy_entity_set:
    print(entity)

print("\nEntities in both SpaCy and NLTK:")
for entity in spacy_entity_set & nltk_entity_set:
    print(entity)
