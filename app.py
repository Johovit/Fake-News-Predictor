from flask import Flask, render_template, request
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords once
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer

model = pickle.load(open('models/fake_news_model.pkl', 'rb'))

vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# Text preprocessing

ps = PorterStemmer()

stop_words = stopwords.words('english')

def clean_text(text):

    text = text.lower()

    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub(r'\n', '', text)

    text = re.sub(r'\w*\d\w*', '', text)

    words = text.split()

    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# Home route

@app.route('/')
def home():
    return render_template('index.html')

# Prediction route

@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    cleaned_news = clean_text(news)

    vector_input = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector_input)[0]

    if prediction == 0:
        result = "FAKE NEWS"
    else:
        result = "REAL NEWS"

    return render_template(
        'result.html',
        prediction=result,
        news_text=news
    )

if __name__ == '__main__':
    app.run(debug=True)