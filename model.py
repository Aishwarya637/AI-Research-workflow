import streamlit as st
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Research Workflow", layout="wide")
st.title("🎓 Topic Research Synthesizer")
st.subheader("Analyze peer-reviewed literature on any topic instantly")

st.sidebar.header("Workflow Settings")
num_papers = st.sidebar.slider("Number of papers to fetch", min_value=3, max_value=15, value=5)

# --- STEP 1: OPEN-ACCESS ARXIV INTEGRATION ---
def fetch_arxiv_papers(query, limit=5):
    """Fetches relevant papers from the free ArXiv API."""
    # Format query safely for URL
    formatted_query = query.strip().replace(" ", "+")
    base_url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&max_results={limit}"
    
    try:
        response = urllib.request.urlopen(base_url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        papers = []
        namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', namespaces):
            title = entry.find('atom:title', namespaces).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', namespaces).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', namespaces).text[:4]  # Extract year
            url = entry.find('atom:id', namespaces).text
            
            papers.append({
                "title": title,
                "year": published,
                "abstract": summary,
                "url": url,
                "citationCount": "N/A"
            })
        return papers
    except Exception as e:
        st.error(f"Failed to connect to ArXiv: {e}")
        return []

# --- MAIN UI WORKFLOW ---
# We make the text box empty by default and guide the user via the placeholder
query_input = st.text_input(
    label="What topic would you like to research today?",
    value="",
    placeholder="e.g., impact of social media on students, quantum computing in cryptography, microplastics in ocean life..."
)

if st.button("Execute Research Workflow", type="primary"):
    # If the user clicks search without typing anything, assign a backup search topic
    search_topic = query_input.strip()
    if not search_topic:
        search_topic = "impact of social media on students"
        st.info(f"💡 No topic entered. Defaulting search to: **'{search_topic}'**")

    if not openai_api_key:
        st.error("🔑 Missing OpenAI API Key! Please create a `.env` file in your root folder containing `OPENAI_API_KEY=\"your_key\"`.")
    else:
        # Initialize LLM & Embeddings dynamically using the API key from .env
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini", temperature=0.2)
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)

        with st.spinner(f"Step 1: Querying open-access papers for '{search_topic}'..."):
            papers = fetch_arxiv_papers(search_topic, limit=num_papers)
        
        if not papers:
            st.warning("No papers found. Try adjusting your search queries.")
        else:
            st.success(f"Successfully retrieved {len(papers)} papers!")
            
            # Display retrieved papers
            st.markdown("### 📚 Source Literature")
            paper_data = []
            documents_for_rag = []
            
            for p in papers:
                title = p.get("title", "N/A")
                year = p.get("year", "N/A")
                abstract = p.get("abstract", "No abstract available.")
                url = p.get("url", "#")
                
                paper_data.append({
                    "Title": title,
                    "Year": year,
                    "URL": url
                })
                
                text_content = f"Title: {title}\nYear: {year}\nAbstract: {abstract}"
                documents_for_rag.append(text_content)
            
            df = pd.DataFrame(paper_data)
            st.dataframe(df, use_container_width=True)
            
            # --- STEP 2: EMBEDDINGS & VECTOR STORE (FAISS) ---
            with st.spinner("Step 2: Vectorizing Abstracts and Building Knowledge Base..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                split_docs = text_splitter.create_documents(documents_for_rag)
                
                db = FAISS.from_documents(split_docs, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 4})
            
            # --- STEP 3: LLM SYNTHESIS & REPORT GENERATION ---
            with st.spinner("Step 3: Generating Synthesized Executive Summary..."):
                relevant_chunks = retriever.invoke(search_topic)
                context_str = "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])
                
                # We update the system prompt template to use the dynamically entered subject
                prompt = ChatPromptTemplate.from_messages([
                    ("system", (
                        "You are an elite academic research assistant. Synthesize a comprehensive "
                        "literature review based ONLY on the provided research paper fragments below.\n\n"
                        f"RESEARCH DATA:\n{context_str}"
                    )),
                    ("human", (
                        f"Synthesize a report detailing: '{search_topic}'. "
                        "Break the report down into three key logical sections/themes based "
                        "on the nature of the academic findings returned in the research data. "
                        "Include citations using the authors/years noted in the research data. "
                        "Keep your analysis balanced, rigorous, and direct."
                    ))
                ])
                
                chain = prompt | llm
                synthesis = chain.invoke({})
                
                st.markdown("---")
                st.markdown(f"### 📝 Synthesized Executive Report: {search_topic.title()}")
                st.markdown(synthesis.content)