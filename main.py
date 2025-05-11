from preprocessing import clean_text
from features import extract_features
from model import train_and_evaluate

with open('dataset/author_a.txt', 'r') as f:
    a_text = f.read()
with open('dataset/author_b.txt', 'r') as f:
    b_text = f.read()
with open('dataset/disputed.txt', 'r') as f:
    disputed_text = f.read()

a_clean = clean_text(a_text)
b_clean = clean_text(b_text)
disputed_clean = clean_text(disputed_text)

texts = [' '.join(a_clean), ' '.join(b_clean)]
labels = ['Author A', 'Author B']

vectorizer = extract_features(texts)
X = vectorizer.fit_transform(texts)
model = train_and_evaluate(X, labels)

disputed_vec = vectorizer.transform([' '.join(disputed_clean)])
prediction = model.predict(disputed_vec)
print("Predicted Author of disputed manuscript:", prediction[0])
