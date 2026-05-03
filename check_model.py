from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели
model_path = "./model_datasphere_dataset_4000"
tokenizer = AutoTokenizer.from_pretrained("alexyalunin/RuBioRoBERTa")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

id2label = {0: "нарушение_диеты", 1: "прием_лекарств"}

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=48)
    with torch.no_grad():
        logits = model(**inputs).logits
    return id2label[logits.argmax().item()]

# Тестовые фразы и ожидаемые метки
test_data = [
    # нарушение_диеты (10 фраз)
    ("Я выпил чашку кофе за час до анализа", "нарушение_диеты"),
    ("Съел яблоко утром перед сдачей крови", "нарушение_диеты"),
    ("Позавтракал бутербродом за два часа", "нарушение_диеты"),
    ("Выпил стакан сока, когда проснулся", "нарушение_диеты"),
    ("Съел шоколадку по дороге в поликлинику", "нарушение_диеты"),
    ("Пил чай с сахаром за два часа до визита", "нарушение_диеты"),
    ("Съел печенье, не удержался", "нарушение_диеты"),
    ("Выпил молоко за три часа до сдачи", "нарушение_диеты"),
    ("Пил газировку утром", "нарушение_диеты"),
    ("Съел булочку перед выходом из дома", "нарушение_диеты"),
    # прием_лекарств (10 фраз)
    ("Принимал антибиотик накануне вечером", "прием_лекарств"),
    ("Выпил таблетку от давления утром", "прием_лекарств"),
    ("Пью противовоспалительное уже три дня", "прием_лекарств"),
    ("Принимал обезболивающее ночью", "прием_лекарств"),
    ("Выпил антигистаминное перед сном", "прием_лекарств"),
    ("Принимаю статины ежедневно, сегодня тоже пил", "прием_лекарств"),
    ("Выпил витамины с железом утром", "прием_лекарств"),
    ("Принимал парацетамол от температуры вчера", "прием_лекарств"),
    ("Выпил мочегонное средство утром", "прием_лекарств"),
    ("Принимал сорбенты за два часа до анализа", "прием_лекарств"),
]

correct = 0
total = len(test_data)

print("Результаты тестирования:\n")
for text, expected in test_data:
    predicted = predict(text)
    is_correct = (predicted == expected)
    if is_correct:
        correct += 1
    print(f"Фраза: {text}")
    print(f"Ожидалось: {expected} → Получено: {predicted} | {'✓' if is_correct else '✗'}\n")

accuracy = correct / total * 100
print(f"Точность (accuracy): {correct}/{total} = {accuracy:.2f}%")