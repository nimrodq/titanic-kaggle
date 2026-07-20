import pandas as pd


def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()

    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royalty",
        "Countess": "Royalty",
        "Sir": "Royalty",
        "Don": "Royalty",
        "Dona": "Royalty",
        "Jonkheer": "Royalty",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Professional",
        "Rev": "Professional"
    }

    return title_map.get(title, title)


def family_group(size):
    if size == 1:
        return "Solo"
    elif size <= 3:
        return "Small"
    elif size <= 5:
        return "Medium"
    return "Large"


def fill_age(df):
    age_lookup = (
        df.groupby(["Sex", "Pclass", "Title"])["Age"]
        .median()
    )

    overall_median = df["Age"].median()

    def impute(row):
        if pd.isna(row["Age"]):
            key = (row["Sex"], row["Pclass"], row["Title"])
            if key in age_lookup.index:
                return age_lookup.loc[key]
            return overall_median
        return row["Age"]

    df["Age"] = df.apply(impute, axis=1)
    return df


def preprocess_data(train, test):
    train = train.copy()
    test = test.copy()

    train["is_train"] = 1
    test["is_train"] = 0

    combined = pd.concat([train, test], ignore_index=True, sort=False)

    combined["Title"] = combined["Name"].apply(extract_title)

    combined["FamilySize"] = (
        combined["SibSp"] +
        combined["Parch"] +
        1
    )

    combined["IsAlone"] = (
        combined["FamilySize"] == 1
    ).astype(int)

    combined["FamilyGroup"] = combined["FamilySize"].apply(family_group)

    combined["Fare"] = combined["Fare"].fillna(
        combined["Fare"].median()
    )

    combined["Embarked"] = combined["Embarked"].fillna(
        combined["Embarked"].mode()[0]
    )

    combined = fill_age(combined)

    combined["FarePerPerson"] = (
        combined["Fare"] /
        combined["FamilySize"]
    )

    combined["TicketPrefix"] = (
        combined["Ticket"]
        .str.replace(r"[0-9]", "", regex=True)
        .str.replace(r"[./]", "", regex=True)
        .str.replace(" ", "")
    )

    combined["TicketPrefix"] = combined["TicketPrefix"].replace(
        "",
        "NONE"
    )

    train = (
        combined[combined["is_train"] == 1]
        .drop(columns=["is_train"])
        .reset_index(drop=True)
    )

    test = (
        combined[combined["is_train"] == 0]
        .drop(columns=["is_train", "Survived"])
        .reset_index(drop=True)
    )

    return train, test


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    train, test = preprocess_data(train, test)

    print(train.head())