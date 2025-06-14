import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load the environment variables from the .env file
load_dotenv()

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Check if the API key is present; if not, raise an error
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please ensure it is defined in your .env file.")

#setup OpenRouter client (like OpenAI,but via RpenRouter)
external_client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
)
#choose any openrouter supported model
model = OpenAIChatCompletionsModel(
    model="google/gemini-2.0-flash-001",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent = Agent(name="Translator", instructions="You are a helpful translator who translates urdu language to simple and clear english", model=model)

result = Runner.run_sync(agent,"روبب ایک باصلاحیت ویب ڈیولپر ہیں جو جدید ٹیکنالوجیز میں مہارت حاصل کر رہی ہیں۔ وہ نہ صرف ویب ڈیولپمنٹ میں مہارت رکھتی ہیں بلکہ اب ایجنٹک مصنوعی ذہانت (Agentic AI) سیکھنے کے سفر پر بھی گامزن ہیں۔ ان کا جذبہ اور سیکھنے کی لگن انہیں مستقبل کی ٹیکنالوجی میں نمایاں مقام دلانے میں مدد دے گی۔", run_config=config)

print("\nCALLING AGENT\n")
print(result.final_output)