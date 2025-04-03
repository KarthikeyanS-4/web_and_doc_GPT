"""
RAG System with Supabase and GPT-4o
"""
import os
import re
import time
import ast
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import pdfplumber
from tqdm import tqdm
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from azure.ai.inference import EmbeddingsClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from supabase import create_client, Client
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://models.inference.ai.azure.com"
model_name = "text-embedding-3-small"
token = os.getenv("AZURE_OPENAI_API_KEY")

client = EmbeddingsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token)
)

cclient = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir='.venv/nltk_data')

class DocumentChunker:
    def __init__(self,
                max_chunk_size: int = 1000,
                overlap: int = 200,
                min_chunk_size: int = 200):

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_by_character(self, text: str) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0

        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            if start > 0:
                start = start - self.overlap
            chunk = {
                "id": chunk_id,
                "text": text[start:end],
                "start_char": start,
                "end_char": end,
                "strategy": "character"
            }
            chunks.append(chunk)
            chunk_id += 1
            start = end

        return chunks

    def chunk_by_paragraph(self, text: str) -> List[Dict[str, Any]]:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_paragraphs = []
        chunk_id = 0
        
        for idx, paragraph in enumerate(paragraphs):
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "paragraphs": current_paragraphs.copy(),
                    "strategy": "paragraph"
                })
                chunk_id += 1
                current_chunk = paragraph
                current_paragraphs = [idx]
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_paragraphs.append(idx)

            if len(paragraph) > self.max_chunk_size:
                sub_chunks = self.chunk_by_character(paragraph)
                for sub_chunk in sub_chunks:
                    sub_chunk["parent_paragraph"] = idx
                    sub_chunk["strategy"] = "paragraph+character"
                    sub_chunk["id"] = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
                current_chunk = ""
                current_paragraphs = []

        if current_chunk:
            chunks.append({
                "id": chunk_id,
                "text": current_chunk,
                "paragraphs": current_paragraphs,
                "strategy": "paragraph"
            })

        return chunks

    def chunk_by_semantic_coherence(self, text: str) -> List[Dict[str, Any]]:
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = ""
        current_sentences = []
        chunk_id = 0

        for idx, sentence in enumerate(sentences):
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk,
                    "sentence_indices": current_sentences.copy(),
                    "strategy": "semantic"
                })
                chunk_id += 1
                if current_sentences and len(current_sentences) > 1:
                    overlap_sentence_idx = current_sentences[-1]
                    overlap_sentence = sentences[overlap_sentence_idx]
                    current_chunk = overlap_sentence + " " + sentence
                    current_sentences = [overlap_sentence_idx, idx]
                else:
                    current_chunk = sentence
                    current_sentences = [idx]
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_sentences.append(idx)

            if len(sentence) > self.max_chunk_size:
                sub_chunks = self.chunk_by_character(sentence)
                for sub_chunk in sub_chunks:
                    sub_chunk["parent_sentence"] = idx
                    sub_chunk["strategy"] = "semantic+character"
                    sub_chunk["id"] = chunk_id
                    chunks.append(sub_chunk)
                    chunk_id += 1
                current_chunk = ""
                current_sentences = []

        if current_chunk:
            chunks.append({
                "id": chunk_id,
                "text": current_chunk,
                "sentence_indices": current_sentences,
                "strategy": "semantic"
            })

        return chunks

    def chunk_pdf_with_metadata(self, pdf) -> List[Dict[str, Any]]:
        all_chunks = []
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_chunks = self.chunk_by_paragraph(page_text)
            for chunk in page_chunks:
                chunk["page_num"] = page_num
                chunk["total_pages"] = len(pdf.pages)
                if hasattr(page, "page_number"):
                    chunk["page_number"] = page.page_number
                chunk["page_width"] = page.width
                chunk["page_height"] = page.height
                all_chunks.append(chunk)
        return all_chunks

    def chunk_website_with_metadata(self, soup) -> List[Dict[str, Any]]:
        all_chunks = []
        sections = []
        headings = soup.find_all(['h1', 'h2', 'h3'])
        for heading in headings:
            section_content = ""
            section_title = heading.get_text().strip()
            section_tag = heading.name
            current = heading.next_sibling
            while current:
                if current.name in ['h1', 'h2', 'h3'] and current.name <= heading.name:
                    break
                if hasattr(current, 'get_text'):
                    section_content += current.get_text() + " "
                current = current.next_sibling
            if section_content:
                sections.append({
                    "title": section_title,
                    "level": section_tag,
                    "content": section_content.strip()
                })
        if not sections:
            paragraphs = soup.find_all("p")
            sections.append({
                "title": soup.title.get_text() if soup.title else "Untitled",
                "level": "p",
                "content": "\n".join([p.get_text() for p in paragraphs])
            })
        for section_idx, section in enumerate(sections):
            section_chunks = self.chunk_by_semantic_coherence(section["content"])
            for chunk in section_chunks:
                chunk["section_title"] = section["title"]
                chunk["section_level"] = section["level"]
                chunk["section_idx"] = section_idx
                chunk["total_sections"] = len(sections)
                all_chunks.append(chunk)
        return all_chunks


class EmbeddingProcessor:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 10) -> List[Dict[str, Any]]:
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk["text"] for chunk in batch]
            response = self.client.embed(
                input=batch_texts,
                model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            for j, embedding in enumerate(embeddings):
                chunks[i+j]["embedding"] = embedding
        return chunks

    def save_to_supabase(self, chunks: List[Dict[str, Any]], document_id: str, document_name: str):
        for i, chunk in enumerate(chunks):
            embedding = chunk.get("embedding", [])
            row_data = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_id": chunk["id"],
                "chunk_index": i,
                "content": chunk["text"],
                "embedding": embedding,
                "metadata": {k: v for k, v in chunk.items() 
                             if k not in ["id", "text", "embedding"]}
            }
            supabase.table("document_chunks").insert(row_data).execute()

    def clear_supabase_data(self):
        try:
            response = supabase.table("document_chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            if response:
                print("Successfully cleared the 'document_chunks' table.")
            else:
                print(f"Failed to clear the table: {response.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error clearing data in Supabase: {e}")

    def query_similar_chunks(self, query_text: str, similarity_threshold: float = 0.60, max_chunks: int = 10):
        query_response = self.client.embed(
            input=[query_text],
            model=self.model_name
        )
        query_embedding = query_response.data[0].embedding
        response = supabase.table("document_chunks").select("chunk_id, content, embedding, metadata, document_name").execute()
        all_chunks = response.data
        similar_chunks = []
        for chunk in all_chunks:
            chunk_embedding = chunk.get("embedding", [])
            if isinstance(chunk_embedding, str):
                chunk_embedding = ast.literal_eval(chunk_embedding)
            if chunk_embedding:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0]
                if similarity >= similarity_threshold:
                    similar_chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {}),
                        "document_name": chunk.get("document_name", ""),
                        "similarity": similarity
                    })
        similar_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_chunks[:max_chunks]


class RAGSystem:
    def __init__(self, embedding_processor):
        self.embedding_processor = embedding_processor
    
    def answer_question(self, question: str, similarity_threshold: float = 0.60):
        relevant_chunks = self.embedding_processor.query_similar_chunks(
            query_text=question,
            similarity_threshold=similarity_threshold
        )
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        context = ""
        sources = []
        for chunk in relevant_chunks:
            context += f"\nChunk (Similarity: {float(chunk['similarity']):.2f}):\n{chunk['content']}\n"
            source_info = {
                "document": chunk.get("document_name", "Unknown"),
                "chunk_id": chunk.get("chunk_id", "Unknown"),
                "similarity": float(chunk["similarity"]),
                "text": chunk["content"]
            }
            metadata = chunk.get("metadata", {})
            if metadata:
                if "page_number" in metadata:
                    source_info["page"] = metadata["page_num"]
                if "total_pages" in metadata:
                    source_info["total_pages"] = metadata["total_pages"]
                if "section_title" in metadata:
                    source_info["section_title"] = metadata["section_title"]

            sources.append(source_info)
        system_prompt = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
        If the context doesn't contain relevant information to answer the question, say "I don't have enough information to answer this question."
        Don't make up information. If you know additional information about the topic but it's not in the context, don't include it.
        Cite sources by referring to the chunk numbers when appropriate."""
        user_prompt = f"""Question: {question}
                                Context:
                                {context}

                                Please provide a clear and concise answer based solely on the information in the context."""
        try:
            response = cclient.complete(
                messages=[
                    SystemMessage(system_prompt),
                    UserMessage(user_prompt),
                ],
                temperature=0.5,
                top_p=0.5,
                model="gpt-4o"
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources
            }


def extract_text_from_pdf(file):
    chunker = DocumentChunker(max_chunk_size=1000, overlap=200)
    with pdfplumber.open(file) as pdf:
        chunks = chunker.chunk_pdf_with_metadata(pdf)
    return chunks

def scrape_website(url):
    chunker = DocumentChunker(max_chunk_size=2000, overlap=200)
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    chunks = chunker.chunk_website_with_metadata(soup)
    return chunks

def setup_supabase_schema():
    try:
        response = supabase.table("document_chunks").select("count", count="exact").limit(1).execute()
        print("Table exists, skipping creation")
    except:
        print("Creating document_chunks table")
        sql = """
        create extension if not exists vector;
        create table if not exists document_chunks (
            id uuid primary key default uuid_generate_v4(),
            document_id text not null,
            document_name text not null,
            chunk_id integer not null,
            chunk_index integer not null,
            content text not null,
            embedding vector(1536),
            metadata jsonb,
            created_at timestamptz default now()
        );
        create index on document_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
        """
        supabase.rpc("execute_sql", {"sql": sql}).execute()
        print("Table created successfully")

def streamlit_integration():
    st.set_page_config(
        page_title="RAG System with Supabase & GPT-4o",
        page_icon="📚",
        layout="wide"
    )
    chunker = DocumentChunker(max_chunk_size=1000, overlap=200)
    processor = EmbeddingProcessor(client, model_name)
    rag_system = RAGSystem(processor)
    if "schema_initialized" not in st.session_state:
        with st.spinner("Setting up database..."):
            try:
                setup_supabase_schema()
                st.session_state.schema_initialized = True
            except Exception as e:
                st.error(f"Error setting up database: {str(e)}")
    st.title("📚 RAG System with Supabase & GPT-4o")
    tab1, tab2 = st.tabs(["Process Documents", "Ask Questions"])
    with tab1:
        st.header("Process Documents")
        option = st.selectbox("Choose input method", ["Upload PDF", "Provide URL"])
        if option == "Upload PDF":
            uploaded_files = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)
            chunking_method = st.selectbox("Choose chunking strategy", 
                                        ["By Paragraph", "By Semantic Coherence", "By Character"])
            
            if uploaded_files:
                process_button = st.button("Process and Store Document")
                
                if process_button:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    # processor.clear_supabase_data()
                    total_chunks = 0
                    all_doc_sources = {}  # Track document sources
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        progress_text.write(f"Processing {uploaded_file.name} ({i}/{len(uploaded_files)})...")
                        doc_id = f"pdf_{int(time.time())}_{i}"
                        doc_name = uploaded_file.name
                        
                        all_doc_sources[doc_id] = doc_name  # Store document source info
                        
                        with st.spinner("Processing PDF..."):
                            with pdfplumber.open(uploaded_file) as pdf:
                                if chunking_method == "By Paragraph":
                                    chunks = chunker.chunk_pdf_with_metadata(pdf)
                                elif chunking_method == "By Semantic Coherence":
                                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                                    semantic_chunks = chunker.chunk_by_semantic_coherence(text)
                                    
                                    page_char_counts = [len(page.extract_text() or "") for page in pdf.pages]
                                    total_chars = 0
                                    
                                    for chunk in semantic_chunks:
                                        if "start_char" in chunk:
                                            for page_num, page_chars in enumerate(page_char_counts):
                                                if total_chars + page_chars >= chunk["start_char"]:
                                                    chunk["page_num"] = page_num
                                                    break
                                                total_chars += page_chars
                                            else:
                                                chunk["page_num"] = len(pdf.pages) - 1
                                        chunk["total_pages"] = len(pdf.pages)
                                    chunks = semantic_chunks
                                else:
                                    text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                                    chunks = chunker.chunk_by_character(text)
                                    for chunk in chunks:
                                        chunk["total_pages"] = len(pdf.pages)
                                        
                            for chunk in chunks:
                                chunk["source_id"] = doc_id
                                chunk["source_name"] = doc_name
                            
                            with st.status("Generating embeddings and storing in Supabase"):
                                chunks_with_embeddings = processor.generate_embeddings(chunks)
                                processor.save_to_supabase(chunks_with_embeddings, doc_id, doc_name)
                                total_chunks += len(chunks)
                                st.write(f"✅ Successfully processed {len(chunks)} chunks")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if 'document_sources' not in st.session_state:
                        st.session_state.document_sources = {}
                    st.session_state.document_sources.update(all_doc_sources)
                    
                    progress_text.empty()
                    progress_bar.empty()
                    st.success(f"PDF processed successfully! Created and stored {total_chunks} chunks in Supabase.")
                    
                    if chunks:
                        with st.expander("Sample chunk"):
                            sample = chunks[0]
                            st.write(f"Chunk ID: {sample['id']}")
                            st.write(f"Strategy: {sample['strategy']}")
                            st.write(f"Source: {sample['source_name']}")
                            if 'page_num' in sample:
                                st.write(f"Page: {sample['page_num'] + 1} of {sample['total_pages']}")
                            st.text_area("Text", sample['text'][:500])

        
        elif option == "Provide URL":
            url = st.text_input("Enter Website URL")
            chunking_method = st.selectbox("Choose chunking strategy", 
                                          ["By HTML Structure", "By Semantic Coherence", "By Paragraph"])
            
            if st.button("Scrape and Store Website"):
                if url:
                    doc_id = f"url_{int(time.time())}"
                    doc_name = url
                    
                    with st.spinner("Processing website..."):
                        response = requests.get(url)
                        soup = BeautifulSoup(response.text, "html.parser")
                        
                        if chunking_method == "By HTML Structure":
                            chunks = chunker.chunk_website_with_metadata(soup)
                        elif chunking_method == "By Semantic Coherence":
                            text = soup.get_text()
                            chunks = chunker.chunk_by_semantic_coherence(text)
                            for chunk in chunks:
                                chunk["url"] = url
                                chunk["title"] = soup.title.get_text() if soup.title else "Untitled"
                        else:
                            paragraphs = soup.find_all("p")
                            text = "\n\n".join([p.get_text() for p in paragraphs])
                            chunks = chunker.chunk_by_paragraph(text)
                            for chunk in chunks:
                                chunk["url"] = url
                                chunk["title"] = soup.title.get_text() if soup.title else "Untitled"
                        
                        with st.status("Generating embeddings and storing in Supabase"):
                            chunks_with_embeddings = processor.generate_embeddings(chunks)
                            
                            # processor.clear_supabase_data()
                            
                            processor.save_to_supabase(chunks_with_embeddings, doc_id, doc_name)
                            
                            st.write(f"✅ Successfully processed {len(chunks)} chunks")
                        
                        st.success(f"Website scraped successfully! Created and stored {len(chunks)} chunks in Supabase.")
                        
                        if chunks:
                            with st.expander("Sample chunk"):
                                sample = chunks[0]
                                st.write(f"Chunk ID: {sample['id']}")
                                st.write(f"Strategy: {sample['strategy']}")
                                if 'section_title' in sample:
                                    st.write(f"Section: {sample['section_title']}")
                                st.text_area("Text", sample['text'][:500])
                else:
                    st.error("Please enter a valid URL.")
    
    with tab2:
        st.header("Ask Questions")
        
        try:
            response = supabase.table("document_chunks").select("document_id, document_name").execute()
            docs = response.data
            unique_docs = {}
            for doc in docs:
                doc_id = doc["document_id"]
                doc_name = doc["document_name"]
                unique_docs[doc_id] = doc_name
            
            if not unique_docs:
                st.info("No documents found. Please process and store a document first.")
            else:
                st.write(f"🗃️ Available documents: {len(unique_docs)}")
                for doc_id, doc_name in unique_docs.items():
                    st.write(f"- {doc_name}")
                
                question = st.text_input("Ask a question about your documents")
                
                with st.expander("Advanced Settings"):
                    similarity_threshold = st.slider(
                        "Similarity Threshold", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.60, 
                        step=0.05,
                        help="Minimum cosine similarity score (higher = more relevant)"
                    )

            col1, col2 = st.columns([9, 1])  # Adjust the ratio as needed
            
            with col1:
                if st.button("Search and Answer") and question:
                    with st.spinner("Searching for relevant information..."):
                        result = rag_system.answer_question(
                            question=question,
                            similarity_threshold=similarity_threshold
                        )
                        
                        st.markdown("### Answer")
                        st.markdown(result["answer"])
                        if result["sources"]:
                            st.markdown("### Sources")
                            for i, source in enumerate(result["sources"]):
                                with st.expander(f"Source {i+1} - {source['document']} (Similarity: {source['similarity']:.2f})"):
                                    if "page" in source:
                                        st.write(f"📄 Page: {source['page']} of {source['total_pages']}")
                                    if "url" in source:
                                        st.write(f"🔗 URL: {source['url']}")
                                    if "section_title" in source:
                                        st.write(f"📝 Section: {source['section_title']}")
                                    # Show text content
                                    st.markdown("#### Content")
                                    st.markdown(source['text'])
            
            with col2:
                if st.button("Clear Supabase Data"):
                    with st.spinner("Clearing data..."):
                        processor.clear_supabase_data()
                        st.success("Supabase data cleared successfully.")
                                
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")

if __name__ == "__main__":
    streamlit_integration()
