import pandas as pd


def preprocess_data(train, test):

    train = train.copy()
    test = test.copy()

    combined = pd.concat([train, test])

    for df in [combined]:

        df["Title"] = df["Name"].str.extract(
            " ([A-Za-z]+)\.",
            expand=False
        )

        df["Title"] = df["Title"].replace(
            [
                "Lady",
                "Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer"
            ],
            "Rare"
        )

        df["Title"] = df["Title"].replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs"
            }
        )

        df["FamilySize"] = (
            df["SibSp"] +
            df["Parch"] +
            1
        )

        df["IsAlone"] = 0
        df.loc[df["FamilySize"] == 1, "IsAlone"] = 1


    def fill_age(row):
        if pd.isnull(row["Age"]):
            return combined.groupby(
                ["Sex", "Pclass"]
            )["Age"].median()[
                row["Sex"],
                row["Pclass"]
            ]
        return row["Age"]

    combined["Age"] = combined.apply(
        fill_age,
        axis=1
    )
    combined["Fare"] = combined["Fare"].fillna(
        combined["Fare"].median()
    )
    for index, row in combined.iterrows():

        if pd.isnull(row["Age"]):

            combined.loc[index, "Age"] = age_medians[
                row["Sex"],
                row["Pclass"],
                row["Title"]
            ]


    train = combined.iloc[:len(train)]
    test = combined.iloc[len(train):]

    return train, test