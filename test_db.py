import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path="./vector_db", settings=Settings(anonymized_telemetry=False, allow_reset=False))
print(client.list_collections())
