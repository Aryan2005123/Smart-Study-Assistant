# EduNova AI – Smart Study Assistant v2.0

A Streamlit learning workspace with document upload, NLP retrieval, optional transformer-powered question answering and summarization, automatic revision notes, quiz generation, activity history and learning analytics.

## Features
- Modern dashboard with learning KPIs and quick actions.
- PDF library: upload text-based PDFs and extract text with PyMuPDF.
- Demo study guide PDF and demo text/CSV datasets included.
- AI Study Tutor:
  - Fast NLP mode: TF-IDF + cosine similarity passage retrieval.
  - Deep Learning mode: DistilBERT question-answering pipeline.
- Smart Notes:
  - Offline extractive NLP summarization.
  - Optional DistilBART transformer summarization.
- Quiz Center: creates fill-in-the-blank MCQs from the active document, grades answers and stores attempts.
- Learning Analytics: quiz accuracy, activity distribution and attempt history.
- SQLite database stores document metadata, quiz attempts and activity locally.

## Setup (Windows)
1. Extract the ZIP.
2. Open the extracted `EduNova_AI_v2` folder in VS Code.
3. Open Terminal in that folder.
4. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Run:

   ```powershell
   streamlit run app.py
   ```

Open the local URL shown in the terminal (usually http://localhost:8501).

## First run and transformer models
The Fast NLP and extractive summary modes work without downloading transformer weights. Selecting DistilBERT QA or DistilBART summarization downloads the corresponding pretrained model the first time and may require several GB of disk space/RAM depending on the PyTorch build and model cache. A working internet connection is required for the first download. If model loading fails, the app reports the issue and uses the built-in NLP fallback.

## Included demo assets
- `sample_docs/EduNova_AI_Sample_Study_Guide.pdf` – upload this through My Library.
- `data/demo_study_material.txt` – loaded by the Load demo study material button.
- `data/demo_courses.csv` – sample course progress dataset.
- `data/demo_study_activity.csv` – sample study activity dataset.
- `data/demo_quiz_questions.csv` – sample quiz question dataset.

## Important technical notes
- The transformer pipelines are genuine pretrained deep-learning models. They are optional/lazy-loaded; their weights are not bundled in this ZIP.
- The app's automatic quiz generator is a lightweight heuristic MCQ generator, not a trained question-generation model.
- PDF text extraction supports selectable text. Scanned PDFs require OCR, which is not included.
- The Fast NLP tutor retrieves relevant passages; it does not generate new answers. Use the DistilBERT QA mode for model-based extractive answers.
- This project is a student prototype. Validate important answers against the original course material.

## Troubleshooting
- If `streamlit` is not recognized, activate `.venv` and run `python -m streamlit run app.py`.
- If PyMuPDF installation fails, use Python 3.11 and reinstall with `pip install PyMuPDF`.
- If transformer model loading runs out of memory, use Fast NLP mode and Extractive NLP notes.
- If a PDF uploads but no text is extracted, it may be scanned/image-only.
