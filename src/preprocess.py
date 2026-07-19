import pandas as pd


def preprocess_data(train, test):

    train = train.copy()
    test = test.copy()

    for df in [train, test]:

        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        df["IsAlone"] = 0
        df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

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

        df["Deck"] = df["Cabin"].str[0]
        df["Deck"] = df["Deck"].fillna("U")

    return train, test