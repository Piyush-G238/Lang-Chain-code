from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries.
From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.
These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth.
Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=70,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

index = 0
for chunk in chunks:
    print(index+1, chunk)
    index += 1