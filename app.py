import os
import subprocess
import time
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import whisper
import nltk
from nltk.tokenize import word_tokenize
from flask_bootstrap import Bootstrap5
import random

# --- NLTK Resource Check and Download ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- Flask App Initialization ---
app = Flask(__name__)
Bootstrap5(app)
app.config['SECRET_KEY'] = 'a_secure_key_for_session_management'

# --- Folders ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {
    'UPLOADS': os.path.join(BASE_DIR, 'uploads'),
    'AUDIO': os.path.join(BASE_DIR, 'audio'),
    'TRANSCRIPTS': os.path.join(BASE_DIR, 'transcripts'),
    'RESULTS': os.path.join(BASE_DIR, 'results'),
    'STATIC': os.path.join(BASE_DIR, 'static')
}
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# --- State & Model Files ---
STATE_FILE = os.path.join(FOLDERS['RESULTS'], 'app_state.pkl')
MODEL_FILE = os.path.join(FOLDERS['RESULTS'], 'lr_model.pkl')
VECTORIZER_FILE = os.path.join(FOLDERS['RESULTS'], 'vectorizer.pkl')

# --- Globals ---
CATEGORIES = []
REAL_KEYWORDS_DATA = {}
MODEL = None
VECTORIZER = None
TRAINING_METRICS = {}
CONFUSION_MATRIX = {}
PREDICTION_HISTORY = []
APP_INITIALIZED = False

# --- Dataset Path ---
DATASET_PATH = os.path.join(BASE_DIR, 'keywords.xlsx')

# --- Utility Functions ---
def load_keywords_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_excel(path)
    cols = {c: c.strip().lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    cat_col = next((c for c in ['category', 'cat', 'label', 'class'] if c in df.columns), None)
    kw_col = next((c for c in ['keyword', 'word', 'key', 'term'] if c in df.columns), None)

    if not cat_col or not kw_col:
        raise ValueError("Dataset must contain category and keyword columns.")

    df[cat_col] = df[cat_col].astype(str).str.strip()
    df[kw_col] = df[kw_col].astype(str).str.strip().str.lower()
    grouped = df.groupby(cat_col)[kw_col].apply(lambda s: list(dict.fromkeys(s))).to_dict()
    return grouped

def load_state():
    global CONFUSION_MATRIX, PREDICTION_HISTORY, TRAINING_METRICS
    initial_cm = {cat: {pred: 0 for pred in CATEGORIES} for cat in CATEGORIES}
    initial_metrics = {k: 0.0 for k in ['training_accuracy', 'testing_accuracy', 'precision', 'recall', 'f1_score']}
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'rb') as f:
                state = pickle.load(f)
                CONFUSION_MATRIX = state.get('cm', initial_cm)
                PREDICTION_HISTORY = state.get('history', [])
                TRAINING_METRICS = state.get('metrics', initial_metrics)
                return
    except Exception as e:
        print(f"Error loading state: {e}")
    CONFUSION_MATRIX = initial_cm
    PREDICTION_HISTORY = []
    TRAINING_METRICS = initial_metrics

def save_state():
    state = {
        'cm': CONFUSION_MATRIX,
        'history': PREDICTION_HISTORY,
        'metrics': TRAINING_METRICS
    }
    try:
        with open(STATE_FILE, 'wb') as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving state: {e}")

# --- Realistic Synthetic Data Generation ---
FILLER_WORDS = ["the", "is", "on", "with", "and", "a", "an", "in", "at", "for", "by", "this", "that", "of"]

def create_training_dataframe_from_keywords(keywords_dict):
    data, labels = [], []
    np.random.seed(42)
    for cat, kws in keywords_dict.items():
        n_docs = max(50, len(kws) * 2)  # More sentences per category
        for _ in range(n_docs):
            sample_kws = np.random.choice(kws, size=min(6, len(kws)), replace=False)
            # Add 2-4 filler words randomly
            filler = np.random.choice(FILLER_WORDS, size=random.randint(2,4), replace=True)
            sentence_words = list(sample_kws) + list(filler)
            random.shuffle(sentence_words)
            sentence = " ".join(sentence_words)
            data.append(sentence)
            labels.append(cat)
    df = pd.DataFrame({'text': data, 'label': labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# --- Model Training ---
def train_model(df):
    global MODEL, VECTORIZER, TRAINING_METRICS
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    VECTORIZER = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = VECTORIZER.fit_transform(X_train)
    X_test_vec = VECTORIZER.transform(X_test)

    MODEL = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    MODEL.fit(X_train_vec, y_train)

    y_pred_train = MODEL.predict(X_train_vec)
    y_pred_test = MODEL.predict(X_test_vec)

    TRAINING_METRICS['training_accuracy'] = float(accuracy_score(y_train, y_pred_train))
    TRAINING_METRICS['testing_accuracy'] = float(accuracy_score(y_test, y_pred_test))
    TRAINING_METRICS['precision'] = float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0))
    TRAINING_METRICS['recall'] = float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0))
    TRAINING_METRICS['f1_score'] = float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))

    with open(MODEL_FILE, 'wb') as f: pickle.dump(MODEL, f)
    with open(VECTORIZER_FILE, 'wb') as f: pickle.dump(VECTORIZER, f)
    save_state()

# --- Audio Extraction and Transcription ---
def extract_audio(video_path, audio_path):
    command = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '0', audio_path, '-y']
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return False

def get_transcript(audio_path, transcript_path):
    model = whisper.load_model("base.en")
    result = model.transcribe(audio_path, fp16=False)
    transcript = result.get("text", "")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    return transcript

# --- Keyword Matching & Prediction ---
def highlight_and_count_keywords(transcript, keywords_data):
    words = word_tokenize(transcript.lower())
    match_counts = {cat: 0 for cat in CATEGORIES}
    keyword_to_categories = {}
    for cat, kw_list in keywords_data.items():
        for kw in kw_list:
            keyword_to_categories.setdefault(kw.strip().lower(), set()).add(cat)

    highlighted_parts = []
    for w in words:
        cleaned = w.strip('.,!?"\'()[]:;').lower()
        if cleaned in keyword_to_categories:
            for cat in keyword_to_categories[cleaned]:
                match_counts[cat] += 1
            highlighted_parts.append(f'<span class="highlight">{w}</span>')
        else:
            highlighted_parts.append(w)
    return ' '.join(highlighted_parts), match_counts

def predict_category(transcript):
    if MODEL is None or VECTORIZER is None:
        return None, None, "Model or Vectorizer not initialized."
    try:
        vec = VECTORIZER.transform([transcript])
        pred = MODEL.predict(vec)[0]
        probs = MODEL.predict_proba(vec)[0]
        prob_dict = {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))}
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        return pred, sorted_probs, None
    except Exception as e:
        return None, None, f"Prediction failed: {e}"

# --- Startup Tasks ---
def startup_tasks():
    global REAL_KEYWORDS_DATA, CATEGORIES, MODEL, VECTORIZER
    try:
        REAL_KEYWORDS_DATA = load_keywords_dataset()
        CATEGORIES = list(REAL_KEYWORDS_DATA.keys())
        print(f"Loaded categories: {CATEGORIES}")
    except Exception as e:
        print(f"Dataset load error: {e}")
        return

    load_state()
    if set(CONFUSION_MATRIX.keys()) != set(CATEGORIES):
        CONFUSION_MATRIX.clear()
        for cat in CATEGORIES:
            CONFUSION_MATRIX[cat] = {pred: 0 for pred in CATEGORIES}
        PREDICTION_HISTORY.clear()
        save_state()

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, 'rb') as f: MODEL = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f: VECTORIZER = pickle.load(f)
        print("Loaded persisted model & vectorizer.")
    else:
        print("Training new model...")
        df = create_training_dataframe_from_keywords(REAL_KEYWORDS_DATA)
        train_model(df)
        print("Model trained.")

@app.before_request
def startup_tasks_check():
    global APP_INITIALIZED
    if not APP_INITIALIZED:
        startup_tasks()
        APP_INITIALIZED = True

# --- Flask Routes ---
@app.route('/')
def upload_page():
    return render_template('index.html', total_videos=len(PREDICTION_HISTORY), metrics=TRAINING_METRICS, categories=CATEGORIES)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    global CONFUSION_MATRIX, PREDICTION_HISTORY
    error = None
    video_url = audio_url = transcript = highlighted_transcript = None
    match_counts = {}
    predicted_category = None
    sorted_probs = None

    if 'video_file' not in request.files:
        error = "No video file uploaded."
        return render_template('result.html', error=error)

    file = request.files['video_file']
    true_label = request.form.get('true_label')

    if file.filename == '':
        error = "No file selected."
        return render_template('result.html', error=error)

    filename = secure_filename(file.filename)
    base_name = os.path.splitext(filename)[0]
    timestamp = int(time.time())
    video_path = os.path.join(FOLDERS['UPLOADS'], f"{base_name}_{timestamp}{os.path.splitext(filename)[1]}")
    audio_path = os.path.join(FOLDERS['AUDIO'], f"{base_name}_{timestamp}.mp3")
    transcript_path = os.path.join(FOLDERS['TRANSCRIPTS'], f"{base_name}_{timestamp}.txt")
    file.save(video_path)
    video_url = url_for('audio_files', filename=os.path.basename(video_path))

    if not extract_audio(video_path, audio_path):
        error = "Failed to extract audio."
        return render_template('result.html', error=error)

    audio_url = url_for('audio_files', filename=os.path.basename(audio_path))

    try:
        transcript = get_transcript(audio_path, transcript_path)
    except Exception as e:
        error = f"Transcription failed: {e}"
        return render_template('result.html', error=error)

    highlighted_transcript, match_counts = highlight_and_count_keywords(transcript, REAL_KEYWORDS_DATA)
    predicted_category, sorted_probs, error = predict_category(transcript)

    if true_label and true_label in CATEGORIES and predicted_category:
        CONFUSION_MATRIX[true_label][predicted_category] += 1
        PREDICTION_HISTORY.append((true_label, predicted_category))
        save_state()

    try:
        os.remove(video_path)
    except Exception:
        pass

    return render_template('result.html',
                           error=error,
                           video_url=video_url,
                           audio_url=audio_url,
                           transcript=transcript,
                           highlighted_transcript=highlighted_transcript,
                           match_counts=match_counts,
                           predicted_category=predicted_category,
                           sorted_probs=sorted_probs,
                           total_videos=len(PREDICTION_HISTORY),
                           true_label=true_label,
                           metrics=TRAINING_METRICS,
                           categories=CATEGORIES)

@app.route('/audio/<path:filename>')
def audio_files(filename):
    return send_from_directory(FOLDERS['AUDIO'], filename, as_attachment=False)

@app.route('/get_cm_data')
def get_cm_data():
    cm_display = []
    for true in CATEGORIES:
        row = {'True Label': true}
        total = 0
        for pred in CATEGORIES:
            count = CONFUSION_MATRIX.get(true, {}).get(pred, 0)
            row[pred] = count
            total += count
        row['Total'] = total
        cm_display.append(row)

    cm_metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1_score']}
    if PREDICTION_HISTORY:
        y_true = [t for t, p in PREDICTION_HISTORY]
        y_pred = [p for t, p in PREDICTION_HISTORY]
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm_metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        if labels:
            cm_metrics['precision'] = float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
            cm_metrics['recall'] = float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
            cm_metrics['f1_score'] = float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))

    return jsonify({
        'cm_data': cm_display,
        'categories': CATEGORIES,
        'cm_metrics': cm_metrics,
        'overall_performance_metrics': TRAINING_METRICS,
        'total_videos': len(PREDICTION_HISTORY)
    })

# --- Run App ---
if __name__ == '__main__':
    app.static_folder = FOLDERS['STATIC']
    print("Starting Smart Video Categorization Flask App...")
    app.run(debug=True, host='0.0.0.0', port=5000)
