# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    _import_('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from typing import List, Dict, Optional
import random
from pathlib import Path
import tempfile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datetime import datetime
from transformers import pipeline
import base64

# --- Parsing functions unchanged from before ---

def parse_record_line(line: str) -> Optional[Dict]:
    parts = line.split(';')
    if len(parts) < 7:
        return None
    event = parts[0].strip()
    name = parts[1].strip()
    birth_year = parts[2].strip()
    club = parts[3].strip()
    time = parts[4].strip().strip("'").replace(',', '.')
    date = parts[5].strip().strip("'")
    competition = parts[6].strip()
    location = parts[7].strip() if len(parts) > 7 else ""
    return {
        "event": event,
        "name": name,
        "birth_year": birth_year,
        "club": club,
        "time": time,
        "date": date,
        "competition": competition,
        "location": location
    }

def parse_records(raw_text: str) -> List[Dict]:
    lines = raw_text.strip().split('\n')
    records = []
    for line in lines:
        if line.strip():
            rec = parse_record_line(line)
            if rec:
                records.append(rec)
    return records

def record_to_text(rec: Dict) -> str:
    return (
        f"🏅 **Event:** {rec['event']}\n"
        f"👤 **Athlete:** {rec['name']} (born {rec['birth_year']}), Club: {rec['club']}\n"
        f"⏱️ **Time:** {rec['time']}\n"
        f"📅 **Date:** {rec['date']}\n"
        f"🏟️ **Competition:** {rec['competition']}\n"
        f"📍 **Location:** {rec['location']}\n"
    )

# --- Dummy Vector Store same as before ---

class DummyVectorStore:
    def __init__(self):
        self.documents = []
        self.records = []

    def add_records(self, records: List[Dict]):
        self.records.extend(records)
        self.documents.extend([record_to_text(r) for r in records])

    def search(self, query: str, top_k=5) -> List[str]:
        q_words = set(query.lower().split())
        results = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            if q_words & doc_words:
                results.append(doc)
            if len(results) >= top_k:
                break
        return results

    def filter_by_gender(self, gender: str) -> List[Dict]:
        if gender.lower() == "men":
            return [r for r in self.records if r['name'] and r['birth_year'] and int(r['birth_year']) <= 2005]
        elif gender.lower() == "women":
            return [r for r in self.records if r['name'] and r['birth_year'] and int(r['birth_year']) > 2005]
        else:
            return self.records

# --- Your full raw data for Men and Women records --- #
MEN_RECORDS_RAW = """
50m prosto / 50 free;Jernej Godec;1986;IL;'0:22,19;'11.08.2009;World Military Swimming And...;Montreal (CAN)
100m prosto / 100 free;Anže Tavčar;1994;LL;'0:48,88;'19.05.2016;Evropsko prvenstvo;London (GBR)
200m prosto / 200 free;Sašo Boškan;2002;TK;'1:47,62;'30.11.2023;Rotterdam Qualification Meet;Rotterdam (NED)
400m prosto / 400 free;Martin Bau;1994;BM;'3:50,28;'24.06.2017;Odprto prvenstvo Hrvaške i...;Reka/Rijeka (CRO)
800m prosto / 800 free;Luka Turk;1986;TK;'7:54,58;'27.03.2007;Svetovno prvenstvo;Melbourne (AUS)
1500m prosto / 1500 free;Luka Turk;1986;TK;'15:07,59;'31.03.2007;Svetovno prvenstvo;Melbourne (AUS)

50m prsno / 50 breast;Damir Dugonjić;1988;FR;'0:26,70;'04.08.2015;Svetovno prvenstvo;Kazan (RUS)
100m prsno / 100 breast;Damir Dugonjić;1988;FR;'0:59,66;'26.07.2009;Svetovno prvenstvo;Rim/Rome (ITA)
200m prsno / 200 breast;Matija Može;2000;BP;'2:15,42;'05.07.2018;Mladinsko evropsko prvenstvo;Helsinki (FIN)

50m delfin / 50 fly;Jernej Godec;1986;IL;'0:23,20;'26.07.2009;Svetovno prvenstvo;Rim/Rome (ITA)
100m delfin / 100 fly;Peter Mankoč;1978;IL;'0:51,24;'14.08.2008;Olimpijske igre;Peking/Beijing (CHN)
200m delfin / 200 fly;Robert Žbogar;1989;RAD;'1:57,05;'08.08.2016;Olimpijske igre;Rio de Janeiro (BRA)

50m hrbtno / 50 back;Sašo Boškan;2002;TK;'0:25,80;'10.04.2021;Eindhoven Qualification Meet;Eindhoven (NED)
100m hrbtno / 100 back;Blaž Medvešek;1980;BM;'0:54,88;'25.07.2005;Svetovno prvenstvo;Montreal (CAN)
200m hrbtno / 200 back;Blaž Medvešek;1980;BM;'1:58,61;'20.07.2003;Svetovno prvenstvo;Barcelona (ESP)

200m mešano / 200 I.M.;Anže Ferš Eržen;1999;IL;'2:01,08;'31.07.2023;Prvenstvo za člane, mladin...;Kranj
400m mešano / 400 I.M.;Anže Ferš Eržen;1999;IL;'4:19,63;'02.07.2022;Sredozemske igre;Oran (ALG)
4x50m mešano / 200 medley relay;SLOVENIA;1978;PZS;'1:38,18;'07.08.2009;Absolutno državno prvenstv...;Radovljica
"""

WOMEN_RECORDS_RAW = """
50m prosto / 50 free;Neža Klančar;2000;OL;'0:24,35;'04.08.2024;Olimpijske igre;Pariz/Paris (FRA)
100m prosto / 100 free;Neža Klančar;2000;OL;'0:53,96;'17.04.2024;Australian Open Championships;Southport (AUS)
200m prosto / 200 free;Sara Isaković;1988;RAD;'1:54,97;'13.08.2008;Olimpijske igre;Peking/Beijing (CHN)
400m prosto / 400 free;Anja Klinar;1988;RAD;'4:06,35;'03.07.2016;OPEN de France;Vichy Val dAllier (FRA)
800m prosto / 800 free;Tjaša Oder Prodnik;1994;FR;'8:25,68;'19.05.2016;Evropsko prvenstvo;London (GBR)
1500m prosto / 1500 free;Tjaša Oder Prodnik;1994;FR;'16:08,67;'21.05.2016;Evropsko prvenstvo;London (GBR)

50m prsno / 50 breast;Tara Vovk;2000;LL;'0:31,12;'03.03.2022;2022 TYR Pro Swim Series;Westmont IL (USA)
100m prsno / 100 breast;Tjaša Vozel;1994;ZVK;'1:07,90;'15.10.2020;Odprto združeno prvenstvo ...;Kranj
200m prsno / 200 breast;Tjaša Vozel;1994;ZVK;'2:26,06;'04.10.2019;Svetovni pokal;Budimpešta/Budapest (HUN)

50m delfin / 50 fly;Neža Klančar;2000;OL;'0:25,81;'28.07.2023;Svetovno prvenstvo;Fukuoka (JPN)
100m delfin / 100 fly;Sara Isaković;1988;RAD;'0:58,68;'09.08.2008;Olimpijske igre;Peking/Beijing (CHN)
200m delfin / 200 fly;Sara Isaković;1988;RAD;'2:07,00;'11.07.2009;Svetovno prvenstvo;Rim/Rome (ITA)

50m hrbtno / 50 back;Tina Smrekar;1993;ZVK;'0:29,17;'10.04.2021;Eindhoven Qualification Meet;Eindhoven (NED)
100m hrbtno / 100 back;Tina Smrekar;1993;ZVK;'1:02,25;'25.07.2018;Svetovno prvenstvo;Montreal (CAN)
200m hrbtno / 200 back;Tina Smrekar;1993;ZVK;'2:15,42;'20.07.2017;Svetovno prvenstvo;Barcelona (ESP)

200m mešano / 200 I.M.;Tina Smrekar;1993;ZVK;'2:17,09;'31.07.2023;Prvenstvo za člane, mladin...;Kranj
400m mešano / 400 I.M.;Tina Smrekar;1993;ZVK;'4:40,62;'02.07.2022;Sredozemske igre;Oran (ALG)
"""

# --- Gender-specific record stores ---
men_records = parse_records(MEN_RECORDS_RAW)
women_records = parse_records(WOMEN_RECORDS_RAW)

# --- Ensure equal number of records for stats bar ---
min_count = min(len(men_records), len(women_records))
men_records = men_records[:min_count]
women_records = women_records[:min_count]

men_vector_store = DummyVectorStore()
men_vector_store.add_records(men_records)

women_vector_store = DummyVectorStore()
women_vector_store.add_records(women_records)

all_vector_store = DummyVectorStore()
all_vector_store.add_records(men_records + women_records)

# --- Enhanced CSS for swimming-themed color gradient background, no wallpaper image ---
page_bg_img = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@400;700&display=swap');

.stApp {
    font-family: 'Roboto Slab', serif;
    background: linear-gradient(135deg, #00bcd4 0%, #007ea7 50%, #003459 100%);
    color: #e0f7fa;
    margin: 0;
    padding: 0;
}

/* Container styling with translucent water effect */
section.main > div.block-container {
    background: rgba(3, 169, 244, 0.75); /* bright blue with transparency */
    padding: 2.5rem 3.5rem;
    border-radius: 20px;
    box-shadow: 0 12px 35px rgba(0,105,148,0.7);
    border: 2px solid #00bcd4;
}

/* Buttons style */
.stButton>button {
    background: linear-gradient(45deg, #00bcd4, #00838f);
    color: white;
    font-weight: 700;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 188, 212, 0.6);
}

.stButton>button:hover {
    background: linear-gradient(45deg, #00838f, #00bcd4);
    box-shadow: 0 6px 20px rgba(0, 188, 212, 0.9);
    transform: scale(1.07);
}

/* Sidebar with water ripple effect */
[data-testid="stSidebar"] {
    background: #007ea7;
    color: #e0f7fa;
    font-weight: 600;
    border-right: 2px solid #00bcd4;
}

[data-testid="stSidebar"] h2 {
    font-size: 1.8rem;
    margin-bottom: 10px;
}

/* Sidebar selectbox and textinput styles */
[data-baseweb="select"] > div {
    border-radius: 10px;
    border: 2px solid #00bcd4 !important;
}

[data-baseweb="input"] > div > input {
    border-radius: 10px !important;
    border: 2px solid #00bcd4 !important;
    background: #b2ebf2 !important;
    color: #004d40 !important;
    font-weight: 600;
}

/* Titles with swimming emojis and shadow */
h1, h2, h3 {
    text-shadow: 1px 1px 5px #006064;
}

/* Result cards */
.result-card {
    background: rgba(255, 255, 255, 0.15);
    padding: 15px 20px;
    margin-bottom: 18px;
    border-radius: 15px;
    border: 1.5px solid #00bcd4;
    box-shadow: 0 5px 15px rgba(0, 150, 199, 0.5);
    transition: background 0.3s ease;
}

.result-card:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* Animated wave divider */
@keyframes wave {
    0% { background-position-x: 0; }
    100% { background-position-x: 1000px; }
}

.wave-divider {
    height: 30px;
    background: url('https://s3-us-west-2.amazonaws.com/s.cdpn.io/85486/wave.svg') repeat-x;
    background-size: contain;
    animation: wave 15s linear infinite;
    margin: 30px 0;
}

/* Footer style */
footer {
    font-size: 0.9rem;
    color: #b2ebf2;
    margin-top: 50px;
    padding-top: 10px;
    border-top: 1px solid #00bcd4;
    text-align: center;
}

/* Tooltip style */
.tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted #00bcd4;
    cursor: help;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #00838f;
    color: #e0f7fa;
    text-align: center;
    border-radius: 10px;
    padding: 8px 12px;
    position: absolute;
    z-index: 1;
    bottom: 120%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Page title with emojis and styled header ---
st.markdown("<h1 style='text-align:center;'>🏊‍♂️🇸🇮 Slovenian Swimming Records Q&A 🏅🌊</h1>", unsafe_allow_html=True)

# Friendly intro message with emojis and wave divider
st.markdown("""
<p style='font-size:1.2rem; text-align:center; color:#b2ebf2;'>
Welcome to the <span style='font-weight:bold;'>Slovenian Swimming Records</span> knowledge base! 🏆<br>
Ask me anything about national swimming records for men and women!<br>
Dive in and explore the achievements of 🇸🇮 swimmers! 🌊💧
</p>
""", unsafe_allow_html=True)

# --- Animated swimming emoji above the wave divider (lowered position) ---
st.markdown("""
<div style="position: relative; width: 100%; height: 60px; margin-bottom: -20px; z-index: 10;">
    <div class="swimmer-emoji" style="position: absolute; left: 0; top: 0; width: 100%; height: 60px; pointer-events: none;">
        <span class="swimming-emoji" style="font-size:2.5rem; position: absolute; left: 0; top: 28px;">🏊‍♂️</span>
    </div>
</div>
<style>
@keyframes swim {
    0%   { left: 0;    top: 28px; }
    10%  { left: 10%;  top: 25px; }
    20%  { left: 20%;  top: 31px; }
    30%  { left: 30%;  top: 26px; }
    40%  { left: 40%;  top: 30px; }
    50%  { left: 50%;  top: 28px; }
    60%  { left: 60%;  top: 25px; }
    70%  { left: 70%;  top: 31px; }
    80%  { left: 80%;  top: 26px; }
    90%  { left: 90%;  top: 30px; }
    100% { left: 100%; top: 28px; }
}
.swimming-emoji {
    animation: swim 7s linear infinite;
    transition: left 0.2s, top 0.2s;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="wave-divider"></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 About Plavalna Zveza Slovenije")
st.sidebar.markdown("""
**Plavalna Zveza Slovenije** (Slovenian Swimming Federation) is the national governing body for swimming, water polo, synchronized swimming, and open water swimming in Slovenia. Founded in 1949, it organizes national competitions, supports elite and grassroots athletes, and represents Slovenia in international aquatic sports. 🇸🇮🏊‍♂️

- Official website: [plavalna-zveza.si](https://www.plavalna-zveza.si/)
- Promotes swimming for all ages and abilities
- Hosts national championships and supports Olympic athletes
- Member of LEN and FINA
""")

# Gender selection (no filter button, instant update)
gender_filter = st.sidebar.radio("Select Gender 👤", options=["All", "Men 👨‍🦰", "Women 👩‍🦰"], index=0)

# Query input
query = st.text_input("Ask a question about swimming records 🤔", placeholder="E.g. Who holds the men's 100m freestyle record?")

# Filter records based on sidebar gender selection
if gender_filter == "Men 👨‍🦰":
    filtered_vector_store = men_vector_store
elif gender_filter == "Women 👩‍🦰":
    filtered_vector_store = women_vector_store
else:
    filtered_vector_store = all_vector_store

def answer_query(q: str, vs: DummyVectorStore) -> List[str]:
    results = vs.search(q, top_k=5)
    if not results:
        return ["❌ No relevant records found for your query. Try rephrasing or adjust filters."]
    return results

if query.strip():
    with st.spinner("🏊‍♂️ Diving into records..."):
        answers = answer_query(query, filtered_vector_store)
        st.subheader("🔹 Search Results:")
        for idx, ans in enumerate(answers):
            # Prepend gender label
            gender_label = "Men's Record" if gender_filter == "Men 👨‍🦰" else ("Women's Record" if gender_filter == "Women 👩‍🦰" else "Record")
            emoji = "🏊"
            if "prsno" in ans.lower():
                emoji = "🤽‍♂️"
            elif "delfin" in ans.lower():
                emoji = "🦈"
            elif "hrbtno" in ans.lower():
                emoji = "🌊"
            elif "mešano" in ans.lower() or "medley" in ans.lower():
                emoji = "🎽"
            st.markdown(f'<div class="result-card"><h3>{emoji} {gender_label} #{idx+1}</h3><pre style="white-space: pre-wrap;">{ans}</pre></div>', unsafe_allow_html=True)

# Sidebar example queries with emojis
st.sidebar.markdown("""
### Example Questions ❓
- Who holds the men's 50m freestyle record? 🏅  
- What is the women's 100m backstroke record? 🏊‍♀️  
- Show me 200m breaststroke records for men 🤽‍♂️  
- When was the women's 400m individual medley record set? 🕒  
- List all records for 50m fly 🦈  
""")

# --- Fun facts for sidebar ---
SWIMMING_FUN_FACTS = [
    "The world record for the men's 100m freestyle is under 47 seconds!",
    "Slovenia has produced Olympic swimming finalists and medalists.",
    "Swimming is one of the most popular recreational sports in Slovenia.",
    "The oldest known swimming stroke is the breaststroke.",
    "Competitive swimming became popular in the 19th century."
]

# --- Hero Banner ---
st.markdown("""
<div style='background: linear-gradient(90deg, #00bcd4 60%, #007ea7 100%); border-radius: 18px; padding: 2.5rem 1.5rem 1.5rem 1.5rem; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,105,148,0.18); text-align:center;'>
    <span style='font-size:2.5rem;'>🏊‍♂️🌊</span><br>
    <span style='font-size:2.1rem; font-weight:700; color:#fff;'>Slovenian Swimming Records Q&A</span><br>
    <span style='font-size:1.2rem; color:#b2ebf2;'>Dive into the achievements of Slovenia's best swimmers!</span>
</div>
""", unsafe_allow_html=True)

# --- Animated stats bar ---
def show_stats_bar():
    men_count = len(men_records)
    women_count = len(women_records)
    all_count = men_count + women_count
    st.markdown(f"""
    <div style='display:flex; justify-content:center; gap:2.5rem; margin-bottom:1.5rem;'>
        <div class='metric-card' style='background:rgba(0,188,212,0.13); min-width:120px;'>
            <span style='font-size:2rem;'>👨‍🦰</span><br><b style='font-size:1.3rem;'>{men_count}</b><br><span style='font-size:1rem;'>Men's Records</span>
        </div>
        <div class='metric-card' style='background:rgba(0,188,212,0.13); min-width:120px;'>
            <span style='font-size:2rem;'>👩‍🦰</span><br><b style='font-size:1.3rem;'>{women_count}</b><br><span style='font-size:1rem;'>Women's Records</span>
        </div>
        <div class='metric-card' style='background:rgba(0,188,212,0.13); min-width:120px;'>
            <span style='font-size:2rem;'>🏅</span><br><b style='font-size:1.3rem;'>{all_count}</b><br><span style='font-size:1rem;'>Total Records</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

show_stats_bar()

# Add a fun fact to the sidebar
st.sidebar.markdown(f"<div style='margin-top:20px; padding:10px; background:rgba(0,188,212,0.15); border-radius:10px; color:#004d40; font-size:1.1rem;'><b>🏊 Fun Fact:</b> {random.choice(SWIMMING_FUN_FACTS)}</div>", unsafe_allow_html=True)

# --- Document Q&A Assistant code ---

def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")
    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")
    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")
    raise ValueError(f"Unsupported extension: {ext}")

# --- Add text to ChromaDB ---
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    collection = add_text_to_chromadb.collections[collection_name]
    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )
    return collection

# --- Q&A function ---
def get_answer(collection, question):
    from transformers import pipeline as hf_pipeline
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents."
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    ai_model = hf_pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    response = ai_model(prompt, max_length=150)
    return response[0]['generated_text'].strip()

# --- Enhanced Q&A with source document ---
def get_answer_with_source(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else ["No source"]
    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic.", "No source"
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Answer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    best_source = ids[0].split('_chunk_')[0] if ids else "No source"
    return answer, best_source

# --- Document manager with delete/preview ---
def show_document_manager():
    st.subheader("📋 Manage Documents")
    if not st.session_state.get('converted_docs'):
        st.info("No documents uploaded yet.")
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# --- Search history ---
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    st.subheader("🕒 Recent Searches")
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])

# --- Document statistics ---
def show_document_stats():
    st.subheader("📊 Document Statistics")
    if not st.session_state.get('converted_docs'):
        st.info("No documents to analyze.")
        return
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")

# --- PROFESSIONAL POLISH: Custom CSS and Enhanced UI ---
def add_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: bold;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def safe_convert_files(uploaded_files):
    converted_docs = []
    errors = []
    if not uploaded_files:
        return converted_docs, ["No files uploaded"]
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Converting {uploaded_file.name}...")
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                errors.append(f"{uploaded_file.name}: File too large (max 10MB)")
                continue
            allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in allowed_extensions:
                errors.append(f"{uploaded_file.name}: Unsupported file type")
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                markdown_content = convert_to_markdown(tmp_path)
                if len(markdown_content.strip()) < 10:
                    errors.append(f"{uploaded_file.name}: File appears to be empty or corrupted")
                    continue
                converted_docs.append({
                    'filename': uploaded_file.name,
                    'content': markdown_content,
                    'size': len(uploaded_file.getvalue()),
                    'word_count': len(markdown_content.split())
                })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")
        progress_bar.progress((i + 1) / len(uploaded_files))
    status_text.text("Conversion complete!")
    return converted_docs, errors

def show_conversion_results(converted_docs, errors):
    if converted_docs:
        st.success(f"✅ Successfully converted {len(converted_docs)} documents!")
        total_words = sum(doc['word_count'] for doc in converted_docs)
        st.info(f"📊 Total words added to knowledge base: {total_words:,}")
        with st.expander("📋 View converted files"):
            for doc in converted_docs:
                st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
    if errors:
        st.error(f"❌ {len(errors)} files failed to convert:")
        for error in errors:
            st.write(f"• {error}")

def enhanced_question_interface():
    st.subheader("💬 Ask Your Question")
    with st.expander("💡 Example questions you can ask"):
        st.write("""
        • What are the main topics covered in these documents?
        • Summarize the key points from [document name]
        • What does the document say about [specific topic]?
        • Compare information between documents
        • Find specific data or statistics
        """)
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings in the research paper?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    return question, search_button, clear_button

def check_app_health():
    issues = []
    required_keys = ['converted_docs', 'collection']
    for key in required_keys:
        if key not in st.session_state:
            issues.append(f"Missing session state: {key}")
    try:
        if st.session_state.get('collection'):
            st.session_state.collection.count()
    except Exception as e:
        issues.append(f"Database issue: {e}")
    try:
        pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        issues.append(f"AI model issue: {e}")
    return issues

def show_loading_animation(text="Processing..."):
    with st.spinner(text):
        import time
        time.sleep(0.5)

def add_docs_to_database(collection, converted_docs):
    for doc in converted_docs:
        add_text_to_chromadb(doc['content'], doc['filename'], collection_name=collection.name)
    return len(converted_docs)

# --- ENHANCED MAIN FUNCTION ---
def enhanced_main():
    add_custom_css()
    st.markdown('<h1 class="main-header">📚 Smart Document Knowledge Base</h1>', unsafe_allow_html=True)
    st.markdown("Upload documents, convert them automatically, and ask intelligent questions!")
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'collection' not in st.session_state:
        # Use get_or_create logic to avoid InternalError if collection exists
        client = chromadb.Client()
        try:
            st.session_state.collection = client.create_collection(name="documents")
        except Exception as e:
            # If already exists, get the collection instead
            if 'already exists' in str(e):
                st.session_state.collection = client.get_collection(name="documents")
            else:
                raise
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    health_issues = check_app_health()
    if health_issues:
        with st.expander("⚠️ System Status"):
            for issue in health_issues:
                st.warning(issue)
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "❓ Questions", "📋 Manage", "📊 Analytics"])
    with tab1:
        st.header("📁 Document Upload & Conversion")
        uploaded_files = st.file_uploader(
            "Select documents to add to your knowledge base",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word documents, and text files"
        )
        if st.button("🚀 Convert & Add to Knowledge Base", type="primary"):
            if uploaded_files:
                with st.spinner("Converting documents..."):
                    converted_docs, errors = safe_convert_files(uploaded_files)
                if converted_docs:
                    num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                    st.session_state.converted_docs.extend(converted_docs)
                show_conversion_results(converted_docs, errors)
            else:
                st.warning("Please select files to upload first.")
    with tab2:
        st.header("❓ Ask Questions")
        if st.session_state.converted_docs:
            question, search_button, clear_button = enhanced_question_interface()
            if search_button and question:
                with st.spinner("Searching through your documents..."):
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                st.markdown("### 💡 Answer")
                st.write(answer)
                st.info(f"📄 Source: {source}")
                add_to_search_history(question, answer, source)
            if clear_button:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            if st.session_state.search_history:
                show_search_history()
        else:
            st.info("🔼 Upload some documents first to start asking questions!")
    with tab3:
        show_document_manager()
    with tab4:
        show_document_stats()
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by AI*")

# --- MAIN APP ENTRY ---
if __name__ == "__main__":
    enhanced_main()
