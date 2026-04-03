def chunk_documents(text, chunk_size=1500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period+1]
                end = start + last_period+1
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks