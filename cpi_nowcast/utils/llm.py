import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def get_llm():
    """Initialize local Llama3 via Ollama."""
    return OllamaLLM(model="llama3.2:1b", base_url="http://localhost:11434", temperature=0)

def generate_insight(llm, prompt_data: dict) -> str:
    """Generate narrative insight using Llama3."""
    template = """
You are an expert economist analyzing South Africa CPI data.

{context}

Focus on economic insights, trends, and predictions. Keep response concise (200-300 words).
"""
    prompt = PromptTemplate.from_template(template).format(context=prompt_data["context"])
    
    response = llm.invoke(prompt)
    return response.strip()
