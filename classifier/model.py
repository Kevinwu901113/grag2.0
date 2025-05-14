
import os
import joblib

def load_model(model_dir):
    clf = joblib.load(os.path.join(model_dir, "query_classifier.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return clf, label_encoder

def save_model(clf, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(output_dir, "query_classifier.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

def predict(clf, label_encoder, features):
    pred = clf.predict([features])[0]
    label = label_encoder.inverse_transform([pred])[0]
    return label
