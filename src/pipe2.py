from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
    truncation=True,
)


for t in res:
    print("")
    print(t["generated_text"])
    print("")
