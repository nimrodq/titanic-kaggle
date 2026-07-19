import pandas as pd

def preprocess_data(train, test):
    """
    Clean Titanic data and create features.
    """

    train = train.copy()
    test = test.copy()

    # Create FamilySize
    for df in [train, test]:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Create IsAlone
    for df in [train, test]:
        df["IsAlone"] = 0
        df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # Extract title from name
    for df in [train, test]:
        df["Title"] = df["Name"].str.extract(
            " ([A-Za-z]+)\.",
            expand=False
        )

    return train, test

if __name__ == "__main__":

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train, test = preprocess_data(train, test)

    print(train.head())