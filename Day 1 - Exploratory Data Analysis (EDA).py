# Day 1 - Exploratory Data Analysis (EDA)
# Dataset: Titanic
# Goal: Understand the data using Pandas

import pandas as pd

# Load Titanic dataset directly from URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Basic Info
print("=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== SHAPE (rows, columns) ===")
print(df.shape)

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== BASIC STATISTICS ===")
print(df.describe())

print("\n=== SURVIVAL COUNT ===")
print(df['Survived'].value_counts())
print("0 = Did not survive, 1 = Survived")