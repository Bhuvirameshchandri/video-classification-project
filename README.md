# 🎥 Smart Video Categorization System: Audio Transcript Analysis

## 🎯 Project Overview & Motivation

[cite_start]This project introduces an automated **Machine Learning-based Video Classification System** that categorizes video content by analyzing the **actual spoken words** in the video, providing a reliable alternative to traditional methods that rely on easily manipulated titles or metadata[cite: 1, 17, 19, 283].

[cite_start]The system's core function is to address the serious concern of videos being **misclassified or intentionally mislabeled** (e.g., violent content with innocent titles) to bypass filtering systems[cite: 2, 3, 17].

### Key Categories
[cite_start]The system automatically classifies uploaded videos into five predefined categories based on spoken content[cite: 7, 20, 175]:
* **Auto & Vehicles**
* **Cooking**
* **Education**
* **Pets & Animals**
* **Sports**

### Real-World Applications
[cite_start]The system can be applied in several real-life situations, including[cite: 9, 10, 11, 285]:
* [cite_start]Video platforms (like YouTube) for detecting and preventing wrongly labeled videos[cite: 9].
* [cite_start]Content moderation for identifying potentially harmful or misleading content[cite: 11].
* [cite_start]Parental controls and filtering educational content in schools and libraries[cite: 10].

---

## ⚙️ Core Methodology and Technical Stack

[cite_start]The system processes a raw video through a clear, six-step sequence to convert audio to text and predict the final category[cite: 177, 185]:

### Classification Pipeline
1.  [cite_start]**Video Upload (Flask Interface):** User uploads a video through the simple web interface[cite: 188].
2.  [cite_start]**Audio Extraction (FFmpeg):** The audio track is extracted from the video file and saved in MP3 format[cite: 150, 151, 193, 195].
3.  [cite_start]**Speech-to-Text (OpenAI Whisper):** The audio is converted into an accurate text transcript[cite: 154, 180, 198].
4.  [cite_start]**Text Preprocessing & Keyword Highlighting (NLTK):** The transcript is cleaned (tokenization, stop-word removal) and compared against the keyword dataset[cite: 140, 180, 202, 204].
5.  [cite_start]**Feature Extraction (TF-IDF):** The cleaned text is converted into a numerical matrix using **TF-IDF** (Term Frequency–Inverse Document Frequency)[cite: 181, 208].
6.  [cite_start]**Classification (Logistic Regression):** The TF-IDF features are fed into the trained Logistic Regression model to predict the category with a confidence score[cite: 182, 215, 223].

### Technology Stack
| Component | Tool / Library | Role in the Project |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | [cite_start]Core programming and ML development[cite: 121]. |
| **Speech-to-Text** | **OpenAI Whisper** | [cite_start]State-of-the-art, accurate transcription for real-world videos[cite: 154, 155]. |
| **Classification Model** | **Logistic Regression (Scikit-learn)** | [cite_start]Simple, effective algorithm for text classification with probability predictions[cite: 134, 218, 220, 223]. |
| **Web Interface** | **Flask** | [cite_start]Lightweight framework for the interactive web frontend (upload/results)[cite: 129, 131]. |
| **Audio Utility** | **FFmpeg** | [cite_start]Handles cross-platform audio extraction and conversion[cite: 150, 152]. |
| **NLP Processing** | **NLTK** | [cite_start]Used for text cleaning, tokenization, and stop-word removal[cite: 140, 144]. |

---

## 🛠️ Installation and Setup

### Prerequisites
Ensure the following are installed on your system:
* [cite_start]**Python 3.10** [cite: 111, 121]
* [cite_start]**FFmpeg** (required for audio extraction; must be installed and accessible via system path) [cite: 149]
* [cite_start]**Operating System:** Windows 10 (fully compatible with Linux and macOS)[cite: 111, 117].

### Recommended Hardware
| Component | Requirement |
| :--- | :--- |
| **Processor** | [cite_start]Intel Core i5 [cite: 107] |
| **RAM** | [cite_start]16 GB [cite: 107] |
| **Storage** | [cite_start]256 GB SSD [cite: 107] |

### Quick Start
1.  **Clone the repository:**
    ```bash
    git clone your-repo-url-here
    cd your-repo-name
    ```
2.  **Install dependencies:**
    *(You need to generate a `requirements.txt` file from your Python code's imports)*
    ```bash
    # Install all required Python packages (e.g., Flask, sklearn, whisper, NLTK, pandas)
    pip install -r requirements.txt 
    ```
3.  **Run the Flask application:**
    ```bash
    python [your_main_app_file].py 
    # Example: python app.py
    ```
4.  Access the application in your web browser at the displayed local address (e.g., `http://127.0.0.1:5000/`)[cite: 174].

---

## 📊 Experimental Results and Evaluation

[cite_start]The model was trained on a synthetic dataset (**`keywords.xlsx`**) containing **10,000 entries** for the five categories[cite: 246]. [cite_start]It was tested on a total of **44 diverse video files**[cite: 257].

### Model Performance Metrics
[cite_start]The system achieved the following performance metrics on the testing set[cite: 280]:

| Metric | Result |
| :--- | :--- |
| **Accuracy** | **88.64%** |
| **Precision** | 89.26% |
| **Recall** | 88.64% |
| **F1-Score** | 88.75% |

### Confusion Matrix (Testing on 44 Videos)
| Actual \ Predicted | Autos & Vehicles | Cooking | Education | Pets & Animals | Sports |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Autos & Vehicles (13)** | **11** | 0 | 1 | 1 | 0 |
| **Cooking (11)** | 0 | **11** | 0 | 0 | 0 |
| **Education (11)** | 0 | 2 | **9** | 0 | 0 |
| **Pets & Animals (6)** | 0 | 1 | 0 | **5** | 0 |
| **Sports (3)** | 0 | 0 | 0 | 0 | **3** |
[cite_start]*Correct Predictions: 39/44* [cite: 275, 277]




---

## 🚀 Future Enhancements

[cite_start]The system has several avenues for future improvement[cite: 287, 288, 289, 290]:
* [cite_start]**Multilingual Support:** Incorporating multilingual speech recognition to expand global applicability[cite: 289].
* [cite_start]**Multimodal Analysis:** Combining visual features (frames) with audio analysis to further enhance accuracy[cite: 290].
* [cite_start]**Alerts and Reporting:** Adding real-time alerts for flagged inappropriate content and automatic email reporting for administrators[cite: 288, 289].
