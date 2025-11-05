import os, asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

load_dotenv()

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    assistant = AssistantAgent(
        name="SimpleAgent",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    # Opción A: pasar un string como task
    result = await assistant.run(
        task="Hello, can you introduce yourself?",
        cancellation_token=CancellationToken()
    )
    # El último mensaje suele ser la respuesta del asistente
    print("Agent Response:", result.messages[-1].content)  # type: ignore

    await model_client.close()

asyncio.run(main())
