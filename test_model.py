from transformers import pipeline

# Загружаем модель для задачи классификации текста
print("Загрузка модели RuBioRoBERTa...")
classifier = pipeline("text-classification", model="alexyalunin/RuBioRoBERTa")
print("Модель загружена!")

# Тестовые фразы (позже заменим на реальные вопросы от пациентов)
test_phrases = [
    "Я выпил кофе перед анализом, это нормально?",
    "Вчера я принимал антибиотики, повлияет ли это на результат?",
    "Спасибо, я всё понял."
]

print("\nРезультаты классификации:\n")
for phrase in test_phrases:
    result = classifier(phrase)
    # result — это список, обычно из одного элемента
    label = result[0]['label']
    score = result[0]['score']
    print(f"Фраза: {phrase}")
    print(f"→ Метка: {label}, Уверенность: {score:.4f}\n")