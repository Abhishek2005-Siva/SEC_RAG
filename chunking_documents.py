def chunk_documents(text,chunk_size=500,overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start +chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period+1]
                end = start + last_period+1
        
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

text = ""
with open("sec_filings/aethlon_8ka.txt","r") as f:
    text=f.read()

x = chunk_documents(text)
for i in x:
    print("\n"*5)
    print(i)