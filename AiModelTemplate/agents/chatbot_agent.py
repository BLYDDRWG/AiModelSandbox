from utils.preprocess import preprocess_query
from utils.postprocess import postprocess_response

class ChatbotAgent:
    def __init__(self, retriever, generator, tools):
        self.retriever = retriever
        self.generator = generator
        self.tools = tools

    def handle_query(self, query):
        # Preprocess query
        processed_query = preprocess_query(query)
        
        # Retrieve documents
        documents = self.retriever.retrieve(processed_query)
        
        # Generate response
        response = self.generator.generate(documents)
        
        # Postprocess response
        final_response = postprocess_response(response)
        
        return final_response
