import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Check which columns are empty
columns = df.columns.values
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch', 'Ticket' 'Fare' 'Cabin' 'Embarked']

# Check for NaN values for each column
# age (177), cabin (687), Embarked (2) contain several NaNs
df.isna().sum()

# For embarkment, I gave them C since it had the highest rate of survival females with class 1
# Replace NaN values with reasonable estimate
# df[(df.Survived == 1) & (df.Embarked == "C") & (df.Pclass == 1) & (df.Sex == "female")]

df["Embarked"] = df["Embarked"].fillna(value="C")

# For age, plot class against survival, plot age against survival, plot sex against survival

df[(df.Survived == 1) & (df.Embarked == "C") & (df.Pclass == 1) & (df.Sex == "female")]

df.Pclass.value_counts(normalize=True).plot(kind="bar", alpha = 0.5)

plt.bar(df['Pclass'], df['Survived'])
plt.show() # Depending on whether you use IPython or interactive mode, etc.

df.Embarked.unique()
df.Fare.describe()

# Show me how many survived on each embarkement for missing ages
# if your age was missing, which embarkment where you in ?
df.loc[df.Survived == 1, 'Embarked'].tolist()

df.groupby('Age')['Survived'].nunique()

g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple")