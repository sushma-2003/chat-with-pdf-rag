import os
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# -------------------------
# CONFIG
# -------------------------

PDF_PATH = "pdfs/Test research paper.pdf"

CHUNK_SIZE = 500
TOP_K = 3

# -------------------------
# LOAD API KEY
# -------------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------
# READ PDF
# -------------------------

print("Reading PDF...")

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:

    extracted = page.extract_text()

    if extracted:
        text += extracted + "\n"

print("PDF Loaded")

# -------------------------
# FIXED SIZE CHUNKING
# -------------------------

chunks = [
    text[i:i + CHUNK_SIZE]
    for i in range(0, len(text), CHUNK_SIZE)
]

print(f"Total Chunks Created: {len(chunks)}")

# -------------------------
# EMBEDDING MODEL
# -------------------------

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# CHUNK EMBEDDINGS
# -------------------------

print("Generating Chunk Embeddings...")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("RAG System Ready")

print("\nType 'exit' to quit.\n")

# -------------------------
# CHAT LOOP
# -------------------------

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    # -------------------------
    # QUERY EMBEDDING
    # -------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # -------------------------
    # SIMILARITY SEARCH
    # -------------------------

    similarities = np.dot(
        chunk_embeddings,
        query_embedding
    )

    indices = np.argsort(
        similarities
    )[-TOP_K:][::-1]

    retrieved_chunks = [
        chunks[i]
        for i in indices
    ]

    scores = [
        float(similarities[i])
        for i in indices
    ]

    # -------------------------
    # CONTEXT
    # -------------------------

    context = "\n\n".join(
        retrieved_chunks
    )

    # -------------------------
    # PROMPT
    # -------------------------

    prompt = f"""
Answer ONLY from the context.

If the answer is not present in the context,
reply:

Information not found in retrieved context.

Context:
{context}

Question:
{query}
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # -------------------------
        # DEBUG OUTPUT
        # -------------------------

        print("\n" + "=" * 80)

        print("Question:")
        print(query)

        print("\nRetrieved Chunks:")

        for i, chunk in enumerate(retrieved_chunks):

            print(f"\nChunk {i+1}")
            print("-" * 40)
            print(chunk)

        print("\nSimilarity Scores:")
        print(scores)

        print("\nFinal Answer:")
        print(answer)

        print("\n" + "=" * 80)

    except Exception as e:

        print("\nERROR:")
        print(str(e))