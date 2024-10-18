import time
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://10.23.56.12/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

t1 = time.time()
models = client.models.list()
# model = models.data[0].id
model = "Qwen/LLM"

# # test chat
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    model=model,
    max_tokens=64,
)

result = chat_completion.choices[0].message.content
print("Chat completion output:", result)
t2 = time.time()
print(t2 - t1)

# test completion
completion = client.completions.create(
    model=model,
    prompt="San Francisco is a",
    max_tokens=7,
    temperature=0
)

result = completion.choices[0].text
print("Completion output:", result)
print(time.time() - t2)
