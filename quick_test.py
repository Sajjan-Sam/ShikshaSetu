import os
from openai import OpenAI

# ✅ use your real key here
os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o-mini",              # use a model you have access to
    messages=[{"role": "user", "content": "Say 'ok'"}],
    temperature=0,
)
print(resp.choices[0].message.content)

