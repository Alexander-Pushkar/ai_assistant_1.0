import pandas as pd
import random

df1 = pd.read_csv("нарушение_диеты_2000.csv", sep=';')
df2 = pd.read_csv("прием_лекарств_2000.csv", sep=';')
df = pd.concat([df1, df2], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # перемешиваем
df.to_csv("two_classes_dataset.csv", sep=';', index=False, quoting=1, encoding='utf-8')
print(f"Создан файл two_classes_dataset.csv, всего строк: {len(df)}")