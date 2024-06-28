from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
NUM_RUNS = 10

# Create vector database
def create_vector_db():
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    results = []

    for run in range(1, NUM_RUNS + 1):
        print(f"Run {run} started.")
        run_results = []
        i = 200

        while i <= 5000:
            j = i / 20
            while j <= i / 5:
                startTime = datetime.now()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=i,
                                                               chunk_overlap=j)
                texts = text_splitter.split_documents(documents)
                print(f"Run {run}, Chunk Size: {i}, Overlap: {j} - Documents split.")

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                   model_kwargs={'device': 'cuda'})  # Set to use GPU
                print(f"Run {run}, Chunk Size: {i}, Overlap: {j} - Embeddings model loaded.")

                db = FAISS.from_documents(texts, embeddings)
                db.save_local(DB_FAISS_PATH)
                print(f"Run {run}, Chunk Size: {i}, Overlap: {j} - Vector store created and saved.")

                elapsed_time = datetime.now() - startTime
                print(f"Run {run}, Chunk Size: {i}, Overlap: {j} - Time: {elapsed_time}")

                run_results.append((i, j, elapsed_time.total_seconds()))

                j += i / 20
            i += 50

        # Convert run results to DataFrame and print
        df_run = pd.DataFrame(run_results, columns=['Chunk Size', 'Chunk Overlap', 'Time'])
        print(df_run.to_string(index=False))

        # Add run results to the overall results
        results.extend(run_results)

    return results

def aggregate_results(results):
    df = pd.DataFrame(results, columns=['Chunk Size', 'Chunk Overlap', 'Time'])
    aggregated = df.groupby(['Chunk Size', 'Chunk Overlap']).agg({'Time': 'mean'}).reset_index()
    return aggregated

if __name__ == "__main__":
    print("Chunk Size", "Chunk Overlap", "Time")
    results = create_vector_db()
    aggregated_results = aggregate_results(results)
    print("\nAggregated Results (Average Time in Seconds):\n")
    print(aggregated_results.to_string(index=False))
