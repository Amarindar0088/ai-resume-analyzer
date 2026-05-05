import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stopwords.words('english')
        and word not in string.punctuation
    ]

    return " ".join(filtered_words)