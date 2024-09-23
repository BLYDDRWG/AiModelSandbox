from models.retriever import Retriever
from models.generator import Generator
from models.embeddings import EmbeddingsModel
from agents.chatbot_agent import ChatbotAgent
from tools.tool import Tool
from utils.preprocess import preprocess_query
from utils.postprocess import postprocess_response
import config

def main():
    # Initialize embeddings model
    embeddings_model = EmbeddingsModel(config)
    
    # Initialize retriever
    retriever = Retriever(config, embeddings_model)
    
    # Load prompt template
    with open("templates/prompt_template.txt", "r") as file:
        prompt_template = file.read()
    
    # Initialize generator
    generator = Generator(config, prompt_template)
    
    # Initialize tools
    tools = [Tool(config)]
    
    # Initialize chatbot agent
    agent = ChatbotAgent(retriever, generator, tools)

    while True:
        # Get user input
        query = input("You: ")
        
        # Handle query using the agent
        final_response = agent.handle_query(query)
        
        # Output response
        print(f"Bot: {final_response}")

if __name__ == "__main__":
    main()
