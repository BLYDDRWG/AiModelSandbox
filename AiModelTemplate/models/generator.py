from google.cloud import aiplatform

class Generator:
    def __init__(self, config, prompt_template):
        self.prompt_template = prompt_template
        self.client = aiplatform.gapic.PredictionServiceClient(client_options={"api_endpoint": config.VERTEX_AI_ENDPOINT})
        self.endpoint = config.VERTEX_AI_GENERATION_ENDPOINT

    def generate(self, context):
        prompt = self.prompt_template.format(context=context)
        response = self.client.predict(endpoint=self.endpoint, instances=[{"content": prompt}])
        return response.predictions[0]['content']
