"""Minimal Groq authentication test for the RAG pipeline."""

import os

from llama_index.core.llms import ChatMessage
from llama_index.llms.openai_like import OpenAILike


API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. In PowerShell, run: "
        '$env:GROQ_API_KEY = "your-groq-key"'
    )

print(f"Key length: {len(API_KEY)}")
print(f"Key starts with: {API_KEY[:8]}...")
print(f"Key has stray whitespace: {API_KEY != API_KEY.strip()}")

# Groq provides an OpenAI-compatible Chat Completions endpoint. This small,
# fast model is suitable for validating a free-tier account connection.
llm = OpenAILike(
    model="llama-3.1-8b-instant",
    api_base="https://api.groq.com/openai/v1",
    api_key=API_KEY,
    is_chat_model=True,
    context_window=131072,
    max_tokens=50,
)

try:
    response = llm.chat([ChatMessage(role="user", content="Say hello in 5 words")])
    print("\nSUCCESS:", response)
except Exception as error:
    print("\nFAILED:", type(error).__name__, str(error))
