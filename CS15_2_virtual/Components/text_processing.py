import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')


def process_text(text):
    # Initialize the WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Download the stop words from nltk
    stop_words = set(stopwords.words('english'))

    # Make all the text lowercase
    text = text.lower()

    # Remove the punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove the stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back together
    text = ' '.join(tokens)

    return text