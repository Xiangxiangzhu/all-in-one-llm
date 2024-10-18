import time
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://10.23.56.12/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

t1 = time.time()
models = client.models.list()
model = models.data[0].id

# # test emb
response = client.embeddings.create(
    input="The food was delicious and the waiter...",
    model="Qwen/EMB",
    encoding_format="float"
)

print(response)
print(time.time() - t1)

