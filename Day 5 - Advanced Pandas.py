# Day 5 - Advanced Pandas: groupby, merge, pivot_table
# Goal: Analyse Titanic Dataset like a real Data Analyst
# Tools: Pandas, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("✅ Dataset loaded!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ── Data Cleaning ─────────────────────────────────────────────

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\n✅ Missing values handled!")

# ══════════════════════════════════════════════════════════════
# SKILL 1: GROUPBY
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("SKILL 1: GROUPBY")
print("="*50)

# Survival rate by gender
survival_by_gender = (
    df.groupby('Sex')['Survived']
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print("\nSurvival Rate by Gender:")
print(survival_by_gender)

# Multiple aggregations
class_stats = df.groupby('Pclass').agg(
    avg_age=('Age', 'mean'),
    avg_fare=('Fare', 'mean'),
    passenger_count=('PassengerId', 'count'),
    survival_rate=('Survived', 'mean')
).round(2)

print("\nPassenger Class Statistics:")
print(class_stats)

# Survival by class
survival_by_class = (
    df.groupby('Pclass')['Survived']
    .mean()
    .round(2)
)

print("\nSurvival Rate by Class:")
print(survival_by_class)

# ══════════════════════════════════════════════════════════════
# SKILL 2: PIVOT TABLE
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("SKILL 2: PIVOT TABLE")
print("="*50)

pivot1 = pd.pivot_table(
    df,
    values='Survived',
    index='Sex',
    columns='Pclass',
    aggfunc='mean'
).round(2)

print("\nSurvival Rate by Gender and Class:")
print(pivot1)

pivot2 = pd.pivot_table(
    df,
    values='Fare',
    index='Embarked',
    columns='Pclass',
    aggfunc='mean'
).round(2)

print("\nAverage Fare by Embarked Port and Class:")
print(pivot2)

# ══════════════════════════════════════════════════════════════
# SKILL 3: MERGE
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("SKILL 3: MERGE")
print("="*50)

survival_df = df.groupby('Pclass')['Survived'].mean().reset_index()
survival_df.columns = ['Pclass', 'Survival_Rate']

fare_df = df.groupby('Pclass')['Fare'].mean().reset_index()
fare_df.columns = ['Pclass', 'Average_Fare']

merged = pd.merge(survival_df, fare_df, on='Pclass')

print("\nMerged Survival + Fare Statistics:")
print(merged.round(2))

# ══════════════════════════════════════════════════════════════
# SKILL 4: APPLY + LAMBDA
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("SKILL 4: APPLY + LAMBDA")
print("="*50)

df['Age_Group'] = df['Age'].apply(
    lambda x: 'Child'
    if x < 18
    else ('Adult' if x < 60 else 'Senior')
)

print("\nAge Group Distribution:")
print(df['Age_Group'].value_counts())

# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════

sns.set_style("whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Titanic Advanced Pandas Dashboard',
    fontsize=18,
    fontweight='bold'
)

# Chart 1: Survival by Gender
ax1 = axes[0, 0]
bars = ax1.bar(
    survival_by_gender.index,
    survival_by_gender.values
)

ax1.set_title('Survival Rate by Gender')
ax1.set_ylabel('Survival Rate')

for bar in bars:
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{bar.get_height():.2f}',
        ha='center'
    )

# Chart 2: Survival by Class
ax2 = axes[0, 1]
ax2.bar(
    survival_by_class.index.astype(str),
    survival_by_class.values
)

ax2.set_title('Survival Rate by Passenger Class')
ax2.set_xlabel('Passenger Class')
ax2.set_ylabel('Survival Rate')

# Chart 3: Heatmap
ax3 = axes[1, 0]
sns.heatmap(
    pivot1,
    annot=True,
    fmt='.2f',
    cmap='YlGnBu',
    ax=ax3
)

ax3.set_title('Gender vs Class Survival Heatmap')

# Chart 4: Age Distribution
ax4 = axes[1, 1]

sns.histplot(
    data=df,
    x='Age',
    hue='Survived',
    bins=20,
    ax=ax4
)

ax4.set_title('Age Distribution by Survival')

plt.tight_layout()

plt.savefig(
    'titanic_advanced_dashboard.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\n✅ Dashboard saved as titanic_advanced_dashboard.png")
print("✅ Day 5 Complete!")
print("📌 Concepts Learned:")
print("   - groupby")
print("   - agg")
print("   - pivot_table")
print("   - merge")
print("   - apply + lambda")
print("   - dashboard creation")