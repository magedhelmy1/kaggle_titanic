import pandas as pd
import numpy as np

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


df["Embarked"] = df["Embarked"].fillna(value="C")

# Cleaner way to do the above:
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

cleaned_df['Sex'] = cleaned_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

enbarked_one_hot = pd.get_dummies(cleaned_df['Embarked'], prefix='Embarked')
cleaned_df = cleaned_df.drop('Embarked', axis=1)
cleaned_df = cleaned_df.join(enbarked_one_hot)

x_train = cleaned_df.drop('Survived', axis=1)
y_train = cleaned_df["Survived"].values


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(10, activation="relu", input_dim=x_train.shape[1]),
    Dense(1, activation="sigmoid")

])

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x_train, y_train, epochs=2000)

