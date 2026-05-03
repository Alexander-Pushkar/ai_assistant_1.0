import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch


df = pd.read_csv("two_classes_dataset.csv", sep=';')


le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])
num_labels = len(le.classes_)
print("Метки и их ID:", dict(zip(le.classes_, le.transform(le.classes_))))

# 3. Разделяем на train и test (80% / 20%)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])

# 4. Загружаем токенизатор и модель
model_name = "alexyalunin/RuBioRoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# 5. Создаём класс PyTorch Dataset
class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=32):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 6. Создаём объекты Dataset
train_dataset = MedicalDataset(train_df['text'], train_df['label_id'], tokenizer)
test_dataset = MedicalDataset(test_df['text'], test_df['label_id'], tokenizer)

# 7. Настройки обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",      # исправлено
    save_strategy="epoch",
    save_total_limit=2,         # сохранять только 2 последних чекпоинта
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 8. Метрика (accuracy)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = (predictions == labels).mean()
    return {"accuracy": acc}

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 10. Запуск обучения
trainer.train()

# 11. Сохраняем модель и токенизатор
model.save_pretrained("./saved_model3")
tokenizer.save_pretrained("./saved_model3")
print("Модель сохранена в папку ./saved_model3")

# 12. Оценка на тесте
results = trainer.evaluate()
print(f"Accuracy on test set: {results['eval_accuracy']:.4f}")