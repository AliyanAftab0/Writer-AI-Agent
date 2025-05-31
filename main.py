from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import streamlit as st
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

# Write Agent
writer = Agent(
    name = 'Writer Agent',
    instructions="""You are a helpful writer agent capable of generating essays, stories, poems, and professional emails.
    If someone asks, "Aapko kisne banaya hai?" or "Who created you?", respond with "Mujhe Aliyan ne banaya hai."
    Be professional and creative in your writing tasks."""
)

async def main(user_input):
    return await Runner.run(writer, input=user_input, run_config=config)

st.set_page_config(page_title="Writer Agent", page_icon="✍️", layout="centered")

st.title("Writer Agent ✍️")
st.write("This agent can write poems, stories, essays, emails, and more. It uses the Gemini API to generate text based on your input.")
st.write("Enter your prompt below and click 'Run' to see the generated text.")


user_input = st.text_area(
    "Enter your prompt",
    placeholder="Write a 2 paragraph essay on Generative AI...",
    height=150,
)
st.write("Made with ❤️ by Aliyan")

if st.button("Generate"):
    if user_input.strip() == "":
        raise ValueError("Input cannot be empty. Please enter a prompt.")
    else:
        with st.spinner("Generating..."):
            response = asyncio.run(main(user_input))
            st.success("Generated Response:")
            st.write(response.final_output)
