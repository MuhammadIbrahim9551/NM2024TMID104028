def clean_text(text):
    import re
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return tokens