from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

external_provider = AsyncOpenAI(
    api_key=gemini_api_key,
     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_provider,
)
config = RunConfig(
    model=model,
    model_provider=external_provider,
    tracing_disabled=True,
)
store_agent = Agent(
    name="store_agent",
    instructions="You are a store agent that can assist with health-related inquiries if someone says i am ill and mentions a specific symptom. you will suggest medicines based on the symptom mentioned and explain why.",
   
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the Health Suggester! How can I assist you today?").send()
    
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    # msg = cl.Message(content="")
    # await msg.send()
    
    history.append({"role": "user", "content": message.content})
    result = await Runner.run(
        store_agent,
        input=history,
        run_config=config,
    )
    # async for event in result.events():
        # if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            
        #     await msg.stream_token(event.data.delta)
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    await cl.Message(content=result.final_output).send()
