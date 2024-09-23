import pinecone
from models.embeddings import EmbeddingsModel

class Retriever:
    def __init__(self, config, embeddings_model):
        self.embeddings_model = embeddings_model
        pinecone.init(api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENV)
        self.index = pinecone.Index(config.PINECONE_INDEX_NAME)

    def retrieve(self, query):
        query_vector = self.embeddings_model.embed(query)
        results = self.index.query(query_vector, top_k=config.TOP_K)
        documents = [result['metadata']['text'] for result in results['matches']]
        return documents
