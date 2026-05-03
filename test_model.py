from transformers import pipeline
import pandas as pd


df = pd.read_csv("oak_dataset.csv", sep=';')
print(df.head())
