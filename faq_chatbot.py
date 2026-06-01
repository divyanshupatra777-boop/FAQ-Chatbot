import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download resources (first time only)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# FAQ Data
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return products within 30 days of purchase."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Yes, we offer free shipping on orders above $50."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can contact support at support@example.com."
    }
]

# Stopwords
stop_words = set(stopwords.words('english'))

# Text preprocessing
def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    return " ".join(tokens)

# Prepare data
questions = [preprocess(faq["question"]) for faq in faqs]
answers = [faq["answer"] for faq in faqs]

# Vectorization
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

# Chatbot Function
def chatbot(user_query):

    user_query = preprocess(user_query)

    user_vector = vectorizer.transform([user_query])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    confidence = similarity_scores[0][best_match_index]

    if confidence > 0.2:
        return answers[best_match_index]
    else:
        return "Sorry, I couldn't find a relevant answer."

# Chat Loop
print("FAQ Chatbot")
print("Type 'exit' to quit")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot(query)

    print("Bot:", response)
