from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

X_train = [
    "I've been waiting for a HuggingFace course my whole life.",
    "Python is great!",
]

res = classifier(X_train)
print(res)

batch = tokenizer(
    X_train, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
print(batch)

# with torch.no_grad():
#     outputs = model(**batch)
#     print(outputs)
#     predictions = F.softmax(outputs.logits, dim=1)
#     print(predictions)
#     labels = torch.argmax(predictions, dim=1)
#     print(labels)
