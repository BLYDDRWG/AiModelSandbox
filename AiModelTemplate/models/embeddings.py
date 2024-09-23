from google.cloud import aiplatform

class EmbeddingsModel:
    def __init__(self, config):
        self.client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": config.VERTEX_AI_ENDPOINT})
        self.endpoint = config.VERTEX_AI_EMBEDDINGS_ENDPOINT

    def embed(self, text):
        response = self.client.predict(endpoint=self.endpoint, instances=[{"content": text}])
        return response.predictions[0]['embeddings']
