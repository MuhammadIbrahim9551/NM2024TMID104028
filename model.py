def train_and_evaluate(X, y):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf, X, y, cv=5)
    print("Cross-validation accuracy:", scores.mean())
    clf.fit(X, y)
    return clf