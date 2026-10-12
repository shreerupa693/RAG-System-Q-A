import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the local embedding AI model (lightweight, runs completely on your CPU)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class NPTELVectorDB:
    def __init__(self, db_path="nptel_study.db"):
        # 1. Open a direct connection to a local database file on your hard drive
        self.conn = sqlite3.connect(db_path)
        self.create_table()
        
    def create_table(self):
        with self.conn:
            # 2. Schema definition: Store raw text, week files, page numbers, and vector bytes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    source TEXT,
                    page INTEGER,
                    vector BLOB
                )
            """)
            
    def insert_chunks(self, chunks: list[dict]):
        """Converts text paragraphs into numerical coordinates and stores them."""
        texts = [c["text"] for c in chunks]
        print("\n[AI EMBEDDING] Converting text chunks to numbers using SentenceTransformer...")
        
        # 3. Vector Conversion: Translate English text into multidimensional numeric coordinates
        embeddings = embedding_model.encode(texts)
        
        with self.conn:
            for chunk, vec in zip(chunks, embeddings):
                # Convert the numpy float array list into raw bytes for SQL BLOB storage
                vec_blob = vec.astype(np.float32).tobytes()
                self.conn.execute(
                    "INSERT INTO knowledge_base (text, source, page, vector) VALUES (?, ?, ?, ?)",
                    (chunk["text"], chunk["metadata"]["source_file"], chunk["metadata"]["page_number"], vec_blob)
                )
        print(f"[DATABASE] Saved {len(chunks)} text chunks to your local database successfully.")

    def query_similarity(self, query_text: str, top_k: int = 4) -> list[dict]:
        """Calculates geometric Cosine Similarity to find the matching slide context."""
        # 4. Turn your exam question into a mathematical vector coordinate
        query_vec = embedding_model.encode([query_text]).astype(np.float32)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT text, source, page, vector FROM knowledge_base")
        rows = cursor.fetchall()
        
        results = []
        for text, source, page, vec_blob in rows:
            # Reconstruct the numeric coordinate system out of the database bytes
            db_vec = np.frombuffer(vec_blob, dtype=np.float32)
            
            # 5. Core RAG Math: Geometric Cosine Similarity executed manually
            dot_product = np.dot(query_vec, db_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_d = np.linalg.norm(db_vec)
            similarity = dot_product / (norm_q * norm_d)
            
            results.append({
                "text": text,
                "metadata": {"source": source, "page": page},
                "score": float(similarity.item()) if hasattr(similarity, 'item') else float(similarity[0] if hasattr(similarity, '__len__') else float(similarity))
            })
            
        # 6. Sort results from highest matching percentage to lowest
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
