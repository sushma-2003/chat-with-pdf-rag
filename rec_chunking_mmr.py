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
# EMBEDDING MODEL
# -------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Generating chunk embeddings...")

chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True
)

# -------------------------
# MMR FUNCTION
# -------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

def mmr_search(
    query_embedding,
    document_embeddings,
    k=3,
    lambda_param=0.7
):

    similarities = np.array([
        cosine_similarity(
            query_embedding,
            doc_embedding
        )
        for doc_embedding in document_embeddings
    ])

    selected_indices = []

    first_idx = np.argmax(similarities)

    selected_indices.append(first_idx)

    while len(selected_indices) < k:

        remaining = [
            i
            for i in range(len(document_embeddings))
            if i not in selected_indices
        ]

        mmr_scores = []

        for idx in remaining:

            relevance = similarities[idx]

            diversity = max([
                cosine_similarity(
                    document_embeddings[idx],
                    document_embeddings[selected]
                )
                for selected in selected_indices
            ])

            score = (
                lambda_param * relevance
                -
                (1 - lambda_param) * diversity
            )

            mmr_scores.append(score)

        best_idx = remaining[np.argmax(mmr_scores)]

        selected_indices.append(best_idx)

    return selected_indices, similarities

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
        convert_to_numpy=True
    )

    # -------------------------
    # MMR Retrieval
    # -------------------------

    selected_indices, similarities = mmr_search(
        query_embedding,
        chunk_embeddings,
        k=3,
        lambda_param=0.7
    )

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
            print(chunk[:300])

        print("\nSimilarity Scores:")
        print(scores)

        print("\nFinal Answer:")
        print(answer)

        print("\n" + "=" * 80)

    except Exception as e:

        print("\nERROR:")
        print(str(e))