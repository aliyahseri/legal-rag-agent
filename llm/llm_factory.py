"""
LLM factory for Legal RAG Assistant.
Returns configured LLM instance with Groq/OpenAI fallback.
"""

from config.settings import get_settings
from rich.console import Console

console = Console()


def get_llm():
    """
    Returns the configured LLM instance.
    Uses xAI if XAI_API_KEY is set, Groq if GROQ_API_KEY is set, Gemini if GOOGLE_API_KEY is set, falls back to OpenAI.
    
    Returns:
        Configured LLM instance (ChatOpenAI for xAI/OpenAI, ChatGroq, or ChatGoogleGenerativeAI).
        
    Raises:
        ValueError: If no LLM API key is found.
    """
    settings = get_settings()

    if settings.xai_api_key and settings.primary_llm == "xai":
        from langchain_openai import ChatOpenAI
        console.print(f"[blue]Using xAI LLM: grok-beta[/blue]")
        return ChatOpenAI(
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
            model="grok-beta",
            temperature=0,
        )
    elif settings.groq_api_key and settings.primary_llm == "groq":
        from langchain_groq import ChatGroq
        console.print(f"[blue]Using Groq LLM: {settings.groq_model}[/blue]")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0,
        )
    elif settings.google_api_key and settings.primary_llm == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        console.print(f"[blue]Using Gemini LLM: gemini-2.0-flash[/blue]")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model="gemini-2.0-flash",
            temperature=0,
        )
    elif settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        console.print(f"[blue]Using OpenAI LLM: {settings.primary_llm}[/blue]")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.primary_llm,
            temperature=0,
        )
    else:
        raise ValueError(
            "No LLM API key found. Set XAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY in your .env file."
        )
