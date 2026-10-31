import os, re, json, sqlite3, hashlib
from datetime import datetime, date, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DOC_DIR = APP_DIR / "user_documents"
DB_PATH = APP_DIR / "edunova.db"
DOC_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="EduNova AI | Smart Study Assistant", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f5f7fc; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#101b3c 0%,#1e2b58 100%); }
[data-testid="stSidebar"] * { color: #eef2ff !important; }
[data-testid="stSidebar"] .stRadio label { padding: 9px 12px; border-radius: 10px; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; color: #172554; }
.hero { background: linear-gradient(115deg,#172554,#4f46e5 65%,#7c3aed); color:white; padding:28px 30px; border-radius:20px; margin-bottom:18px; box-shadow:0 12px 30px rgba(49,46,129,.16); }
.hero h1 { color:white; margin:0; font-size:30px; }
.hero p { color:#e0e7ff; margin:7px 0 0 0; }
.metric { background:white; border:1px solid #e7eaf3; padding:18px 20px; border-radius:16px; box-shadow:0 4px 15px rgba(30,41,59,.04); min-height:110px; }
.metric .k { color:#64748b; font-size:13px; font-weight:600; }
.metric .v { color:#172554; font-size:28px; font-weight:800; margin-top:5px; }
.metric .s { color:#64748b; font-size:12px; }
.panel { background:white; padding:20px; border:1px solid #e7eaf3; border-radius:16px; }
.small-muted { color:#64748b; font-size:13px; }
.pill { display:inline-block; padding:5px 10px; border-radius:30px; background:#eef2ff; color:#4338ca; font-size:12px; font-weight:700; }
div.stButton > button { border-radius:10px; font-weight:600; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { background:white; border-radius:9px 9px 0 0; padding:10px 18px; }
</style>
""", unsafe_allow_html=True)

# ---------- Storage ----------
def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, path TEXT, pages INTEGER DEFAULT 0,
        chars INTEGER DEFAULT 0, added_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS quiz_attempts(
        id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT, score INTEGER, total INTEGER, taken_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS activity(
        id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT, detail TEXT, created_at TEXT)""")
    con.commit()
    return con

CON = db()

def log_event(event, detail=""):
    CON.execute("INSERT INTO activity(event,detail,created_at) VALUES(?,?,?)",
                (event, detail, datetime.now().isoformat(timespec="seconds")))
    CON.commit()

def get_docs():
    return pd.read_sql_query("SELECT * FROM documents ORDER BY id DESC", CON)

def get_activity():
    return pd.read_sql_query("SELECT * FROM activity ORDER BY id DESC", CON)

# ---------- Document processing ----------
def extract_pdf(path):
    try:
        import fitz
        doc = fitz.open(path)
        text = "\n".join(page.get_text("text") for page in doc)
        return text, len(doc)
    except ImportError:
        return "", 0
    except Exception as e:
        return "", 0

def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    return text

def sentences(text):
    chunks = re.split(r'(?<=[.!?])\s+', clean_text(text))
    return [x.strip() for x in chunks if len(x.strip()) > 25]

def tokenize(text):
    return re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", str(text).lower())

def fallback_retrieve(question, corpus, top_k=4):
    sents = sentences(corpus)
    if not sents:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        matrix = vect.fit_transform([question] + sents)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        ranked = sorted(zip(scores, sents), reverse=True)[:top_k]
        return [(float(score), sent) for score, sent in ranked if score > 0]
    except Exception:
        q = set(tokenize(question))
        ranked = []
        for sent in sents:
            words = set(tokenize(sent))
            score = len(q & words) / max(1, len(q))
            if score:
                ranked.append((score, sent))
        return sorted(ranked, reverse=True)[:top_k]

def extractive_summary(text, max_sentences=5):
    sents = sentences(text)
    if not sents:
        return "There is not enough readable text to create notes."
    words = tokenize(text)
    freq = {}
    for w in words:
        if w not in {"the","and","for","that","with","this","from","are","was","were","have","has","into","their","then","than","also","you","your","can"}:
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, sent in enumerate(sents):
        score = sum(freq.get(w, 0) for w in tokenize(sent)) / max(1, len(tokenize(sent)))
        scored.append((score, i, sent))
    chosen = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(x[2] for x in sorted(chosen, key=lambda z:z[1]))

@st.cache_resource(show_spinner=False)
def load_transformer_qa():
    from transformers import pipeline
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource(show_spinner=False)
def load_transformer_summary():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def transformer_answer(question, context):
    qa = load_transformer_qa()
    result = qa(question=question, context=context[:12000])
    return result.get("answer", ""), float(result.get("score", 0))

def transformer_summary(text):
    summarizer = load_transformer_summary()
    clipped = text[:10000]
    result = summarizer(clipped, max_length=180, min_length=45, do_sample=False, truncation=True)
    return result[0]["summary_text"]

# ---------- Quiz generation ----------
def build_quiz(text, count=5):
    sents = [s for s in sentences(text) if len(tokenize(s)) >= 7]
    questions = []
    for sent in sents:
        words = [w for w in tokenize(sent) if len(w) > 4]
        if len(words) < 1:
            continue
        answer = max(set(words), key=words.count)
        blanked = re.sub(rf"\b{re.escape(answer)}\b", "________", sent, count=1, flags=re.I)
        distractors = [w for w in set(tokenize(text)) if w != answer and len(w) > 4]
        distractors = sorted(distractors, key=lambda x: (abs(len(x)-len(answer)), x))[:20]
        opts = [answer] + distractors[:3]
        if len(opts) < 4:
            opts += ["concept", "process", "system", "method"]
        opts = list(dict.fromkeys(opts))[:4]
        while len(opts) < 4:
            opts.append("example" + str(len(opts)))
        questions.append({"q": blanked, "answer": answer, "options": opts})
        if len(questions) >= count:
            break
    return questions

def load_sample_text():
    p = DATA_DIR / "demo_study_material.txt"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def init_state():
    defaults = {"corpus": load_sample_text(), "corpus_name": "Demo: Introduction to Artificial Intelligence",
                "quiz": [], "quiz_submitted": False, "quiz_score": None, "last_answer": None, "generated_notes": None, "notes_engine": "NLP"}
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

def hero(title, subtitle, tag="YOUR PERSONAL LEARNING WORKSPACE"):
    st.markdown(f'<div class="hero"><div class="pill">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def metric(label, value, sub):
    st.markdown(f'<div class="metric"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)

# ---------- Sidebar ----------
def set_page(target):
    st.session_state.page_nav = target

with st.sidebar:
    st.markdown("## 🎓 EduNova AI")
    st.caption("SMART STUDY • BETTER RESULTS")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Overview", "My Library", "AI Study Tutor", "Smart Notes", "Quiz Center", "Learning Analytics", "Activity Log"], key="page_nav", label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**CURRENT STUDY MATERIAL**")
    st.caption(st.session_state.corpus_name[:48])
    st.progress(min(len(st.session_state.corpus)/12000, 1.0))
    st.caption(f"{len(st.session_state.corpus):,} characters loaded")
    st.markdown("---")
    st.caption("EduNova AI v2.0 • Local prototype")

# ---------- Overview ----------
if page == "Overview":
    hero("Learn smarter. Every single day.", "Your AI-powered workspace for reading, understanding, revising and tracking progress.")
    docs = get_docs()
    attempts = pd.read_sql_query("SELECT * FROM quiz_attempts", CON)
    activities = get_activity()
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("DOCUMENTS ADDED", len(docs), "PDFs in your library")
    with c2: metric("QUIZ ATTEMPTS", len(attempts), "Practice sessions completed")
    with c3: metric("AVERAGE SCORE", f"{(attempts.score.sum()/max(1,attempts.total.sum())*100):.0f}%", "Across completed quizzes")
    with c4: metric("STUDY ACTIONS", len(activities), "Recorded learning activities")
    st.write("")
    left,right = st.columns([1.35,1])
    with left:
        st.markdown("### 🚀 Quick actions")
        a,b,c = st.columns(3)
        with a:
            st.button("📚 Open library", use_container_width=True, on_click=set_page, args=("My Library",))
        with b:
            st.button("💬 Ask AI tutor", use_container_width=True, on_click=set_page, args=("AI Study Tutor",))
        with c:
            st.button("🧠 Create quiz", use_container_width=True, on_click=set_page, args=("Quiz Center",))
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Recommended study workflow**")
        for n,t in enumerate(["Upload a lecture PDF or open the included demo material","Ask focused questions in the AI Study Tutor","Generate revision notes from the material","Test your understanding in Quiz Center","Review your performance in Learning Analytics"],1):
            st.markdown(f"**{n}.** {t}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("### 📌 Your learning snapshot")
        if len(attempts):
            fig = px.bar(attempts.tail(7), x="taken_at", y="score", color="topic", title="Recent quiz scores", labels={"taken_at":"Attempt","score":"Correct answers"})
            fig.update_layout(height=280, margin=dict(l=10,r=10,t=45,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Your quiz performance will appear here after your first attempt.")
        st.markdown("### ✨ Included in this workspace")
        st.markdown("- PDF library and text extraction\n- Transformer-based Q&A and summarization\n- Automatic revision notes\n- Topic-based practice quizzes\n- Learning analytics and activity history")

# ---------- Library ----------
elif page == "My Library":
    hero("My Study Library", "Bring your course material together in one searchable learning space.", "DOCUMENT MANAGEMENT")
    up_col, sample_col = st.columns([1.2,1])
    with up_col:
        uploaded = st.file_uploader("Upload a study PDF", type=["pdf"], help="Text-based PDFs are supported. Scanned image PDFs need OCR, which is not included.")
        if uploaded and st.button("＋ Add PDF to library", type="primary"):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded.name)
            dest = DOC_DIR / (hashlib.md5(uploaded.getvalue()).hexdigest()[:8] + "_" + safe_name)
            dest.write_bytes(uploaded.getvalue())
            extracted, pages = extract_pdf(dest)
            if not extracted.strip():
                st.error("No selectable text was found. This may be a scanned PDF; try a text-based PDF.")
            else:
                CON.execute("INSERT INTO documents(name,path,pages,chars,added_at) VALUES(?,?,?,?,?)",
                            (uploaded.name, str(dest), pages, len(extracted), datetime.now().isoformat(timespec="seconds")))
                CON.commit()
                st.session_state.corpus = extracted
                st.session_state.corpus_name = uploaded.name
                log_event("PDF uploaded", uploaded.name)
                st.success(f"Added {uploaded.name} • {pages} pages • {len(extracted):,} characters")
    with sample_col:
        st.markdown("#### Try the demo material")
        st.write("Explore the included AI and machine-learning study guide without uploading a file.")
        if st.button("Load demo study material", use_container_width=True):
            st.session_state.corpus = load_sample_text()
            st.session_state.corpus_name = "Demo: Introduction to Artificial Intelligence"
            log_event("Demo material loaded", st.session_state.corpus_name)
            st.success("Demo material loaded. Open AI Study Tutor, Smart Notes or Quiz Center.")
    st.markdown("### Your documents")
    docs = get_docs()
    if docs.empty:
        st.info("No PDFs added yet. Upload a PDF above or use the demo material.")
    else:
        for _, row in docs.iterrows():
            with st.container(border=True):
                x,y,z = st.columns([3,1,1])
                x.markdown(f"**📄 {row['name']}**")
                x.caption(f"Added {row['added_at'][:10]} • {row['chars']:,} characters")
                y.metric("Pages", int(row["pages"]))
                if z.button("Load", key=f"load_{row['id']}"):
                    text, pages = extract_pdf(row["path"])
                    if text.strip():
                        st.session_state.corpus = text
                        st.session_state.corpus_name = row["name"]
                        log_event("Document loaded", row["name"])
                        st.success(f"Loaded {row['name']}")
                        st.rerun()
                    else: st.error("Could not extract text from this PDF.")
    st.caption("Note: uploaded documents are stored locally in the project folder.")

# ---------- AI Tutor ----------
elif page == "AI Study Tutor":
    hero("AI Study Tutor", "Ask questions about your active study material and get context-grounded answers.", "ASK • UNDERSTAND • EXPLORE")
    st.markdown(f"**Active source:** `{st.session_state.corpus_name}`")
    q = st.text_area("What would you like to understand?", placeholder="Example: What is machine learning and how does it differ from traditional programming?", height=100)
    model_choice = st.radio("Answer engine", ["Fast NLP (TF-IDF retrieval)", "Deep Learning (DistilBERT QA)"], horizontal=True)
    if st.button("Generate answer", type="primary", disabled=not q.strip()):
        if model_choice.startswith("Deep"):
            try:
                with st.spinner("Loading DistilBERT model (first use downloads model files)..."):
                    answer, confidence = transformer_answer(q, st.session_state.corpus)
                st.session_state.last_answer = (answer, confidence, "DistilBERT transformer QA")
                log_event("Tutor question", q[:100])
            except Exception as e:
                st.warning(f"Transformer model unavailable: {str(e)[:220]}. Showing the NLP retrieval fallback.")
                hits = fallback_retrieve(q, st.session_state.corpus)
                st.session_state.last_answer = (hits[0][1] if hits else "I could not find a relevant passage in the active material.", hits[0][0] if hits else 0, "TF-IDF fallback")
        else:
            hits = fallback_retrieve(q, st.session_state.corpus)
            if hits:
                answer = "\n\n".join([f"**Relevant passage {i+1}** — {s}" for i,(score,s) in enumerate(hits)])
                st.session_state.last_answer = (answer, hits[0][0], "TF-IDF + cosine similarity")
            else:
                st.session_state.last_answer = ("I could not find a relevant passage. Try using keywords that appear in your document.", 0, "TF-IDF retrieval")
            log_event("Tutor question", q[:100])
    if st.session_state.last_answer:
        answer, confidence, engine = st.session_state.last_answer
        st.markdown("### Answer")
        st.markdown(answer)
        st.caption(f"Engine: {engine} • Relevance/confidence score: {confidence:.2f} (not a calibrated probability)")
        st.info("Answers are grounded in the active document text. Verify important facts against your original study material.")

# ---------- Smart Notes ----------
elif page == "Smart Notes":
    hero("Smart Notes Studio", "Turn long study material into concise revision notes in seconds.", "SUMMARIZE • REVISE")
    st.markdown(f"**Source:** `{st.session_state.corpus_name}`")
    mode = st.radio("Summarization method", ["Extractive NLP (offline)", "Deep Learning (DistilBART transformer)"], horizontal=True)
    max_s = st.slider("Number of key sentences (NLP mode)", 3, 10, 5)
    if st.button("✨ Generate smart notes", type="primary"):
        try:
            with st.spinner("Generating notes..."):
                if mode.startswith("Deep"):
                    notes = transformer_summary(st.session_state.corpus)
                    engine = "DistilBART transformer summarization"
                else:
                    notes = extractive_summary(st.session_state.corpus, max_s)
                    engine = "Extractive sentence scoring"
            st.session_state.generated_notes = notes
            st.session_state.notes_engine = engine
            log_event("Notes generated", engine)
        except Exception as e:
            st.warning(f"Transformer summarizer unavailable: {str(e)[:220]}. Using extractive NLP fallback.")
            st.session_state.generated_notes = extractive_summary(st.session_state.corpus, max_s)
            st.session_state.notes_engine = "Extractive NLP fallback"
    if "generated_notes" in st.session_state:
        st.markdown("### Your revision notes")
        st.markdown(st.session_state.generated_notes)
        st.caption(f"Generated with: {st.session_state.get('notes_engine','NLP')}")
        st.download_button("⬇ Download notes (.txt)", st.session_state.generated_notes,
                           file_name="edunova_smart_notes.txt", mime="text/plain")

# ---------- Quiz ----------
elif page == "Quiz Center":
    hero("Quiz Center", "Check what you know with automatically generated questions from your study material.", "PRACTICE • MEASURE • IMPROVE")
    st.markdown(f"**Quiz source:** `{st.session_state.corpus_name}`")
    n = st.slider("Number of questions", 3, 10, 5)
    if st.button("Generate new quiz", type="primary"):
        st.session_state.quiz = build_quiz(st.session_state.corpus, n)
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = None
        log_event("Quiz generated", f"{len(st.session_state.quiz)} questions")
    if st.session_state.quiz:
        if not st.session_state.quiz_submitted:
            with st.form("quiz_form"):
                answers = []
                for i,item in enumerate(st.session_state.quiz):
                    st.markdown(f"**Q{i+1}. {item['q']}**")
                    answers.append(st.radio("Choose one", item["options"], key=f"quiz_{i}", horizontal=True, label_visibility="collapsed"))
                submitted = st.form_submit_button("Submit answers", type="primary")
            if submitted:
                score = sum(1 for a,item in zip(answers,st.session_state.quiz) if a.lower()==item["answer"].lower())
                st.session_state.quiz_score = score
                st.session_state.quiz_submitted = True
                CON.execute("INSERT INTO quiz_attempts(topic,score,total,taken_at) VALUES(?,?,?,?)",
                            (st.session_state.corpus_name, score, len(st.session_state.quiz), datetime.now().isoformat(timespec="seconds")))
                CON.commit()
                log_event("Quiz submitted", f"{score}/{len(st.session_state.quiz)}")
                st.rerun()
        else:
            score = st.session_state.quiz_score
            total = len(st.session_state.quiz)
            st.success(f"Quiz completed — {score}/{total} correct ({score/total*100:.0f}%)")
            for i,item in enumerate(st.session_state.quiz):
                st.markdown(f"**Q{i+1}. {item['q']}**")
                st.markdown(f"Correct answer: **{item['answer']}**")
            if st.button("Try another quiz"):
                st.session_state.quiz = []
                st.session_state.quiz_submitted = False
                st.rerun()
    else:
        st.info("Generate a quiz to begin. Questions are created from sentences in the active study material.")

# ---------- Analytics ----------
elif page == "Learning Analytics":
    hero("Learning Analytics", "Understand your study consistency and quiz performance.", "MEASURE YOUR PROGRESS")
    attempts = pd.read_sql_query("SELECT * FROM quiz_attempts ORDER BY id", CON)
    activities = get_activity()
    a,b,c = st.columns(3)
    with a: metric("QUIZ ATTEMPTS", len(attempts), "Total completed quizzes")
    with b: metric("QUESTIONS ANSWERED", int(attempts.total.sum()) if len(attempts) else 0, "Across all quiz sessions")
    with c: metric("OVERALL ACCURACY", f"{attempts.score.sum()/max(1,attempts.total.sum())*100:.1f}%" if len(attempts) else "—", "Correct answers / total")
    left,right = st.columns(2)
    with left:
        st.markdown("### Quiz performance")
        if len(attempts):
            chart = attempts.copy()
            chart["attempt"] = range(1,len(chart)+1)
            chart["accuracy"] = chart.score / chart.total * 100
            fig = px.line(chart, x="attempt", y="accuracy", markers=True, range_y=[0,100], title="Accuracy by attempt")
            fig.update_layout(height=330, yaxis_title="Accuracy (%)", xaxis_title="Quiz attempt")
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Complete a quiz to populate your performance chart.")
    with right:
        st.markdown("### Study activity")
        if len(activities):
            counts = activities.groupby("event").size().reset_index(name="count")
            fig = px.pie(counts, names="event", values="count", hole=.55, title="Actions in your workspace")
            fig.update_layout(height=330)
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Your study activity will appear here.")
    if len(attempts):
        st.markdown("### Attempt history")
        st.dataframe(attempts[["topic","score","total","taken_at"]].assign(accuracy=lambda d:(d.score/d.total*100).round(1)), use_container_width=True, hide_index=True)

# ---------- Activity ----------
elif page == "Activity Log":
    hero("Activity Log", "A timeline of your recent learning actions.", "YOUR STUDY HISTORY")
    activities = get_activity()
    if activities.empty: st.info("No activity yet. Upload a PDF, generate notes or take a quiz.")
    else:
        st.dataframe(activities[["created_at","event","detail"]], use_container_width=True, hide_index=True)
    if st.button("Clear activity and quiz history"):
        CON.execute("DELETE FROM activity")
        CON.execute("DELETE FROM quiz_attempts")
        CON.commit()
        st.success("Activity and quiz history cleared.")
        st.rerun()

st.markdown("---")
st.markdown('<div style="text-align:center;color:#94a3b8;font-size:12px;padding:8px">EduNova AI v2.0 • Smart Study Assistant • Built with Streamlit, NLP and Transformer models</div>', unsafe_allow_html=True)
