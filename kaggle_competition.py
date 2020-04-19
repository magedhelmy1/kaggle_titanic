import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

original_df = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Check which columns are empty
columns = original_df.columns.values
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch', 'Ticket' 'Fare' 'Cabin' 'Embarked']

# Check for NaN values for each column
# age (177), cabin (687), Embarked (2) contain several NaNs
original_df.isna().sum()

# drop PassengerId, Name,Cabin (alot of missing data) since they will not contribute to accuracy
df = original_df.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1)

df.isna().sum()

df.describe()
# For embarkment, I gave them C since it had the highest rate of survival females with class 1
# df[(df.Survived == 1) & (df.Embarked == "C") & (df.Pclass == 1) & (df.Sex == "female")]

df["Embarked"] = df["Embarked"].fillna(value="C")

# Some plots
#df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)  ## Setting alpha as per transparency
#plt.show()

# Info on survival
#sum(df.Survived.value_counts())

# when an empty age column column has these values, give it this value:
# df[['Pclass',"Embarked", 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# for a in np.unique(df.Survived):
#     for b in np.unique(df.Pclass):
#         for c in np.unique(df.Sex):
#             for d in np.unique(df.SibSp):
#                 for e in np.unique(df.Parch):
#                     for f in np.unique(df.Embarked):
#                         mean_value = df[(df["Survived"] == a) & (df["Pclass"] == b) & (df["Sex"] == c)
#                                         & (df["SibSp"] == d) & (df["Parch"] == e) & (df["Embarked"] == f)].Age.mean()
#
#                         df = df.assign(Age=np.where(df.Survived.eq(a) & df.Pclass.eq(b) & df.Sex.eq(c) & df.SibSp.eq(d) &
#                                                     df.Parch.eq(e) & df.Embarked.eq(f) & df.Age.isnull(), mean_value, df.Age))

#Cleaner way to do the above:
l_col = ['Survived','Pclass','Sex','Embarked','SibSp','Parch']
df['Age'] = df['Age'].fillna(df.groupby(l_col)['Age'].transform('mean'))

# Examine specific null values
df.isna().sum()
#print(df[df.isnull().any(axis=1)][df.columns.values])

# Specific case with fare 69.5500:

mean_value = df[(df["Survived"] == 0) & (df["Pclass"] == 3)
                & (df["Parch"] == 2) ].Age.mean()

df = df.assign(Age=np.where(df.Survived.eq(0) & df.Pclass.eq(3) & df.SibSp.eq(8)
                            & df.Age.isnull(), mean_value, df.Age))

df.isna().sum()

# Check empty columns
#print(df[df.isnull().any(axis=1)][df.columns.values])

cleaned_df = df.dropna(subset=['Age'])
cleaned_df.isna().sum()
# Switching rep
