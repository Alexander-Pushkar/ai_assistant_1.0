from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#Запуск модели из папки model_datasphere_dataset_4000

model_path = "./model_datasphere_dataset_4000"

tokenizer = AutoTokenizer.from_pretrained("alexyalunin/RuBioRoBERTa")


model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {0: "нарушение_диеты", 1: "прием_лекарств"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
    with torch.no_grad():
        logits = model(**inputs).logits
    return id2label[logits.argmax().item()]

print(predict("проглатил пилюли противоаллергические"))
print(predict("сожрал бургер"))
