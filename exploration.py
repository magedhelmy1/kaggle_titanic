# This is a bunch of code used to explore the dataset


df[(df.Survived == 1) & (df.Embarked == "C") & (df.Pclass == 1) & (df.Sex == "female")]

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


# Some plots
#df.Survived.value_counts(normalize=True).plot(kind="bar", alpha=0.5)  ## Setting alpha as per transparency
#plt.show()

# Info on survival
#sum(df.Survived.value_counts())

# when an empty age column column has these values, give it this value:
# df[['Pclass',"Embarked", 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)