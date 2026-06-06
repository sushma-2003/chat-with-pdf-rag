import os
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------
# CONFIG
# -------------------------

PDF_PATH = "pdfs/Test research paper.pdf"

SIMILARITY_THRESHOLD = 0.50

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

# -------------------------
# RECURSIVE CHUNKING
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

# -------------------------
# EMBEDDINGS
# -------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating embeddings...")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# -------------------------
# CHAT LOOP
# -------------------------

print("\nPDF Chat Ready")
print("Type 'exit' to quit\n")

while True:

    query = input("You: ")

    if query.lower() == "exit":
        break

    # -------------------------
    # Query Embedding
    # -------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # -------------------------
    # COSINE SIMILARITY
    # -------------------------

    similarities = np.dot(
        chunk_embeddings,
        query_embedding
    )

    # -------------------------
    # SCORE THRESHOLD
    # -------------------------

    selected_indices = np.where(
        similarities >= SIMILARITY_THRESHOLD
    )[0]

    # Sort descending
    selected_indices = sorted(
        selected_indices,
        key=lambda x: similarities[x],
        reverse=True
    )

    # Safety limit
    selected_indices = selected_indices[:5]

    if len(selected_indices) == 0:

        print("\nNo relevant chunks found.")
        print("-" * 80)

        continue

    retrieved_chunks = [
        chunks[i]
        for i in selected_indices
    ]

    scores = [
        similarities[i]
        for i in selected_indices
    ]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer only using the provided context.

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