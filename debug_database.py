
import chromadb
import pandas as pd
from pprint import pprint
import os

client = chromadb.PersistentClient(path="./chroma_db")

collections = client.list_collections()

print("\nAvailable collections:")
for col in collections:
    print(col.name)

# ==============================
# EXPORT ALL COLLECTIONS TO CSV
# ==============================
def export_all_collections():
    export_dir = "Database Data"
    os.makedirs(export_dir, exist_ok=True)
    for col in collections:
        print(f"\nProcessing collection: {col.name}")
        collection = client.get_collection(col.name)
        data = collection.get()
        df = pd.DataFrame({
            "id": data["ids"],
            "text": data["documents"],
            "metadata": data["metadatas"]
        })
        # Expand metadata into columns
        meta_df = pd.json_normalize(df["metadata"])
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
        # Save to CSV
        csv_path = os.path.join(export_dir, f"{col.name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported {col.name} to {csv_path}")



# ==============================
# FETCH ALL DATA
# ==============================
# ==============================
# MAIN DEBUG FLOW
# ==============================
def run_debug():
    export_all_collections()

if __name__ == "__main__":
    run_debug()