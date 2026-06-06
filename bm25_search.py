import os
import nltk
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

nltk.download("punkt")
nltk.download("punkt_tab")

# -------------------------
# CONFIG
# -------------------------

PDF_PATH = "pdfs/Test research paper.pdf"

SIMILARITY_BREAKPOINT = 0.75

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
# SENTENCE SPLITTING
# -------------------------

sentences = nltk.sent_tokenize(text)

print(f"Total Sentences: {len(sentences)}")

# -------------------------
# EMBEDDING MODEL
# -------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating Sentence Embeddings...")

sentence_embeddings = embedding_model.encode(
    sentences,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# -------------------------
# SEMANTIC CHUNKING
# -------------------------

chunks = []

current_chunk = [sentences[0]]

for i in range(1, len(sentences)):

    similarity = np.dot(
        sentence_embeddings[i - 1],
        sentence_embeddings[i]
    )

    if similarity < SIMILARITY_BREAKPOINT:

        chunks.append(
            " ".join(current_chunk)
        )

        current_chunk = [sentences[i]]

    else:

        current_chunk.append(
            sentences[i]
        )

if current_chunk:

    chunks.append(
        " ".join(current_chunk)
    )

print(f"Semantic Chunks Created: {len(chunks)}")

# -------------------------
# BM25 INDEX
# -------------------------

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(
    tokenized_chunks
)

print("BM25 Index Created")

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
    # BM25 SEARCH
    # -------------------------

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_k = 3

    indices = np.argsort(
        scores
    )[-top_k:][::-1]

    retrieved_chunks = [
        chunks[i]
        for i in indices
    ]

    retrieved_scores = [
        float(scores[i])
        for i in indices
    ]

    context = "\n\n".join(
        retrieved_chunks
    )

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

        print("\nBM25 Scores:")
        print(retrieved_scores)

        print("\nFinal Answer:")
        print(answer)

        print("\n" + "=" * 80)

    except Exception as e:

        print("\nERROR:")
        print(str(e))