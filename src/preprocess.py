import pandas as pd


def preprocess_data(train, test):

    train = train.copy()
    test = test.copy()


    # ==========================
    # Basic Feature Engineering
    # ==========================
    # Ticket Group Size
    ticket_counts = train["Ticket"].value_counts()

    for df in [train, test]:

        # Surname
        df["Surname"] = (
            df["Name"]
            .str.split(",")
            .str[0]
        )


        # Family Size
        df["FamilySize"] = (
            df["SibSp"] +
            df["Parch"] +
            1
        )


        # Family Size Group
        df["FamilySizeGroup"] = pd.cut(
            df["FamilySize"],
            bins=[
                0,
                1,
                4,
                7,
                20
            ],
            labels=False,
            include_lowest=True
        )


        # Alone
        df["IsAlone"] = (
            df["FamilySize"] == 1
        ).astype(int)


        df["TicketGroupSize"] = (
            df["Ticket"]
            .map(ticket_counts)
            .fillna(1)
        )



        # Title
        df["Title"] = df["Name"].str.extract(
            " ([A-Za-z]+)\\.",
            expand=False
        )


        df["Title"] = df["Title"].replace({

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
            "Dr": "Officer",
            "Rev": "Officer"

        })


        # Cabin information
        df["Deck"] = (
            df["Cabin"]
            .str[0]
            .fillna("U")
        )


        df["CabinKnown"] = (
            df["Cabin"]
            .notna()
            .astype(int)
        )



    # ==========================
    # Surname Feature
    # ==========================

    """ surname_counts = pd.concat(
        [
            train["Surname"],
            test["Surname"]
        ]
    ).value_counts() """


    """ train["SurnameCount"] = (
        train["Surname"]
        .map(surname_counts)
        .fillna(1)
    ) """


    """ test["SurnameCount"] = (
        test["Surname"]
        .map(surname_counts)
        .fillna(1)
    ) """



    # Remove raw surname
    train.drop(
        columns=["Surname"],
        inplace=True
    )

    test.drop(
        columns=["Surname"],
        inplace=True
    )



    # ==========================
    # Age Imputation
    # ==========================

    train["Age"] = train.groupby(
        "Title"
    )["Age"].transform(
        lambda x: x.fillna(
            x.median()
        )
    )


    title_age = (
        train.groupby("Title")["Age"]
        .median()
    )


    test["Age"] = test.apply(

        lambda row:

            title_age[row["Title"]]
            if pd.isna(row["Age"])
            and row["Title"] in title_age.index

            else row["Age"],

        axis=1

    )


    overall_age = train["Age"].median()


    train["Age"] = train["Age"].fillna(
        overall_age
    )


    test["Age"] = test["Age"].fillna(
        overall_age
    )



    # ==========================
    # Woman Child
    # ==========================

    for df in [train, test]:

        # Female or child passengers
        df["WomanChild"] = (
            (df["Sex"] == "female")
            |
            (df["Age"] < 16)
        ).astype(int)

    # ==========================
    # Fare Processing
    # ==========================

    train["Fare"] = train["Fare"].fillna(
        train["Fare"].median()
    )


    test["Fare"] = test["Fare"].fillna(
        train["Fare"].median()
    )


    for df in [train, test]:

        df["FarePerPerson"] = (

            df["Fare"] /
            df["FamilySize"].clip(lower=1)

        )



    # ==========================
    # Embarked
    # ==========================

    train["Embarked"] = train["Embarked"].fillna(
        train["Embarked"].mode()[0]
    )


    test["Embarked"] = test["Embarked"].fillna(
        train["Embarked"].mode()[0]
    )



    # ==========================
    # Age Band
    # ==========================

    for df in [train, test]:

        df["AgeBand"] = pd.cut(

            df["Age"],

            bins=[
                0,
                16,
                32,
                48,
                64,
                100
            ],

            labels=False,

            include_lowest=True

        )



    # ==========================
    # Fare Band
    # ==========================

    train["FareBand"] = pd.qcut(

        train["Fare"],

        4,

        labels=False,

        duplicates="drop"

    )


    fare_bins = train["Fare"].quantile(
        [
            0,
            0.25,
            0.5,
            0.75,
            1
        ]
    ).values



    test["FareBand"] = pd.cut(

        test["Fare"],

        bins=fare_bins,

        labels=False,

        include_lowest=True,

        duplicates="drop"

    )


    test["FareBand"] = test["FareBand"].fillna(
        train["FareBand"].median()
    )



    return train, test



if __name__ == "__main__":

    train = pd.read_csv(
        "data/train.csv"
    )

    test = pd.read_csv(
        "data/test.csv"
    )


    train, test = preprocess_data(
        train,
        test
    )


    print(train.head())
    print(train.columns)