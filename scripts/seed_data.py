"""Seed the database with sample prompts for demo/testing."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import async_session_factory, init_db
from app.models.prompt import Prompt, PromptCategory
from app.models.llm_model import LLMModel


SAMPLE_PROMPTS = [
    {
        "name": "customer_support_response",
        "content": "A customer is emailing about a delayed shipment. They ordered 2 weeks ago and it hasn't arrived. Draft a professional, empathetic response apologizing and explaining the situation.",
        "category": PromptCategory.generation,
        "system_prompt": "You are a professional customer support agent. Be empathetic and helpful.",
        "expected_output": "A polite response acknowledging the delay, apologizing, and providing next steps.",
        "tags": "customer-support,email,production",
    },
    {
        "name": "product_description_summary",
        "content": "Summarize the following product description in 2 sentences: 'Our AI-powered platform provides real-time analytics for e-commerce businesses. It integrates with Shopify, WooCommerce, and Magento. Features include dashboard customization, automated reporting, and inventory forecasting.'",
        "category": PromptCategory.summarization,
        "system_prompt": "You are a concise summarizer. Keep summaries to 2 sentences.",
        "tags": "summarization,e-commerce,production",
    },
    {
        "name": "sentiment_classification",
        "content": "Classify the sentiment of this review as positive, negative, or neutral: 'The product works well but the battery life could be better.'",
        "category": PromptCategory.classification,
        "expected_output": "neutral",
        "tags": "classification,sentiment,testing",
    },
    {
        "name": "code_review_analysis",
        "content": "Review this Python code for potential bugs and style issues:\n\ndef calculate_total(items):\n    total = 0\n    for i in range(len(items)):\n        total = total + items[i].price\n    return total",
        "category": PromptCategory.code,
        "system_prompt": "You are an expert senior software engineer. Review code for bugs, security issues, and style problems.",
        "tags": "code-review,python,testing",
    },
    {
        "name": "financial_qa",
        "content": "What is the difference between a traditional IRA and a Roth IRA? Explain the tax implications of each.",
        "category": PromptCategory.qa,
        "system_prompt": "You are a certified financial advisor. Provide accurate, compliant financial information.",
        "expected_output": "Explanation of traditional IRA (pre-tax contributions, taxed on withdrawal) vs Roth IRA (after-tax contributions, tax-free withdrawals).",
        "tags": "finance,qa,production",
    },
]


SAMPLE_MODELS = [
    {
        "name": "llama3.2",
        "provider": "ollama",
        "description": "Meta's Llama 3.2 - efficient instruction-tuned model",
        "model_type": "open-source",
        "context_window": 8192,
        "gpu_required": True,
        "vram_required_mb": 4096,
    },
    {
        "name": "mistral",
        "provider": "ollama",
        "description": "Mistral 7B - efficient and capable general purpose model",
        "model_type": "open-source",
        "context_window": 8192,
        "gpu_required": True,
        "vram_required_mb": 4096,
    },
]


async def seed():
    await init_db()
    async with async_session_factory() as session:
        for data in SAMPLE_PROMPTS:
            prompt = Prompt(**data)
            session.add(prompt)
            print(f"  Added prompt: {data['name']}")

        for data in SAMPLE_MODELS:
            model = LLMModel(**data)
            session.add(model)
            print(f"  Added model: {data['name']}")

        await session.commit()
    print("Seed complete!")


if __name__ == "__main__":
    asyncio.run(seed())
