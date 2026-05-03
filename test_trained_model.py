import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Абсолютный путь к папке с дообученной моделью
model_path = os.path.abspath("results/checkpoint-20")

# Проверка существования папки
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Папка {model_path} не найдена. Укажите правильный путь.")

# Загружаем токенизатор из оригинальной (недообученной) модели
tokenizer = AutoTokenizer.from_pretrained("alexyalunin/RuBioRoBERTa")

# Загружаем дообученную модель (веса и конфиг)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Соответствие ID меток (такое же, как при обучении)
id2label = {
    0: "жалоба_на_самочувствие",
    1: "нарушение_диеты",
    2: "общий_вопрос",
    3: "прием_лекарств",
    4: "соблюдение_правил"
}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = logits.argmax().item()
    return id2label[pred_id]

# Тестовые фразы
test_phrases = [
    "Я выпил кофе перед анализом, это нормально?",
    "Вчера я принимал антибиотики, повлияет ли это на результат?",
    "Голодал 12 часов, ничего не ел, всё хорошо",
    "У меня болит голова, можно сдавать?",
    "Можно ли пить воду перед сдачей крови?"
]

for phrase in test_phrases:
    label = predict(phrase)
    print(f"Фраза: {phrase}")
    print(f"Предсказанная метка: {label}\n")