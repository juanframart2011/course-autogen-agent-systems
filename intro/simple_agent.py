import os, asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

load_dotenv()

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4.1-mini",  # o "o3-mini" / "gpt-5"
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    agent = AssistantAgent(
        name="SimpleAgent",
        system_message="You are a helpful assistant.",
        model_client=model_client,
    )

    # ✅ Se espera la corutina
    reply = await agent.on_messages(
        [TextMessage(content="Hello, can you introduce yourself?", source="user")],
        CancellationToken()
    )

    print("Agent Response:", reply)
    await model_client.close()

# ✅ Ejecuta la corrutina
asyncio.run(main())
