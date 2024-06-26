from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
NUM_RUNS = 10


# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    results = []

    for run in range(1, NUM_RUNS + 1):
        print(f"Run {run}")
        run_results = []
        i = 200

        while i <= 900:
            j = i / 20
            while j <= i / 5:
                startTime = datetime.now()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=i,
                                                               chunk_overlap=j)
                texts = text_splitter.split_documents(documents)

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                                   model_kwargs={'device': 'cuda'})

                db = FAISS.from_documents(texts, embeddings)
                db.save_local(DB_FAISS_PATH)

                elapsed_time = datetime.now() - startTime

                run_results.append((i, j, elapsed_time.total_seconds()))

                j += i / 20
            i += 150

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
    print("Chunk size", "Chunk Overlap", "Time")
    results = create_vector_db()
    aggregated_results = aggregate_results(results)
    print("\nAggregated Results (Average Time in Seconds):\n")
    print(aggregated_results.to_string(index=False))
