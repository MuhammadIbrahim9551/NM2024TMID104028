def extract_features(text):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    return vectorizer