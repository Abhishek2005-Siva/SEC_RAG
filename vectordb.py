import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("policies")

policies = [
    "Dogs are allowed in the office on Fridays",
    "Pets can come to work on Furry days",
    "Remote work policies allow 3 days from home"
]

for i,policy in enumerate(policies):
    collection.add(
        documents = [policy],
        ids= [f"policy_{i}"]
    )

query = "Can i bring my dog to work"

results = collection.query(query_texts=[query],n_results=2)
print(results)