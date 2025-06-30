# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import streamlit as st
import chromadb
from transformers import pipeline

def setup_documents():
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")

    # Simplified names without Slovenian characters
    my_documents = [
        """Slovenian male swimmers have set impressive national records in freestyle events. For the 50m freestyle, Jernej Godec holds the record at 22.19 seconds, set on August 11, 2009. In the 100m freestyle, Anze Tavcar's record is 48.88 seconds. Saso Boskan holds the 200m freestyle record of 1:47.62. Martin Bau holds the 400m freestyle record at 3:50.28. Luka Turk dominates the 800m and 1500m events with 7:54.58 and 15:07.59.""",

        """In breaststroke, Damir Dugonjic holds the 50m record at 26.70 seconds and the 100m record at 59.66 seconds. Matija Moze holds the 200m record at 2:15.42. These athletes highlight Slovenia’s breaststroke power internationally.""",

        """In butterfly events, Jernej Godec has the 50m record at 23.20 seconds. Peter Mankoc owns the 100m record with 51.24 seconds. Robert Zbogar set the 200m record at 1:57.05. These show Slovenia's speed and endurance in butterfly.""",

        """Backstroke records include Saso Boskan with the 50m backstroke at 25.80 seconds. Blaz Medvesek holds the 100m at 54.88 seconds and the 200m at 1:58.61. These athletes shine in backstroke history.""",

        """The individual medley shows top talent like Anze Fers Erzen, who holds the 200m IM at 2:01.08 and 400m IM at 4:19.63. IM combines all four strokes and reflects versatile Slovenian swimming talent."""
    ]

    collection.add(
        documents=my_documents,
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    return collection

def get_answer(collection, question):
    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    if not docs or min(distances) > 1.5:
        return "🏊 I don't have information about that in my records."

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])

    prompt = f"""Context information:
{context}
Question: {question}
Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.
Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()

    return answer

# --- Streamlit UI ---

st.markdown("## 🏊 Slovenian Swimming Records Q&A")
st.markdown("*Welcome to your personalized swimming records assistant!*")
st.markdown("Ask anything about Slovenia’s national swimming performances (men).")

collection = setup_documents()

st.markdown("### ❓ What do you want to know?")
question = st.text_input("Example: Who holds the 50m backstroke record?")

if st.button("💡 Get My Answer", type="primary"):
    if question:
        with st.spinner("🔎 Searching the swimming archives..."):
            answer = get_answer(collection, question)
        st.success("✅ Here's what I found:")
        st.write(answer)
    else:
        st.info("✍️ Please type a question first.")

with st.expander("ℹ️ About this app"):
    st.markdown("""
    This app was built to answer questions about:
    - National records in Slovenian men's swimming
    - Events like freestyle, backstroke, breaststroke, butterfly, and IM
    - Names, times, and event locations

    **Try asking about:**
    - "Who holds the 100m breaststroke record?"
    - "What is the time for 200m butterfly?"
    - "Where did Luka Turk break the 1500m record?"
    """)

