import pandas as pd


def preprocess_data(train, test):

    train = train.copy()
    test = test.copy()


    # =====================================================
    # Basic Feature Engineering
    # =====================================================

    for df in [train, test]:


        # -----------------------------
        # Family Size Group
        # -----------------------------

        family_size = (
            df["SibSp"] +
            df["Parch"] +
            1
        )


        df["FamilySizeGroup"] = pd.cut(

            family_size,

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



        # -----------------------------
        # Title Extraction
        # -----------------------------

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

            "Dr": "Professional",
            "Rev": "Professional"

        })



        # -----------------------------
        # Cabin Known
        # -----------------------------

        df["CabinKnown"] = (

            df["Cabin"]
            .notna()
            .astype(int)

        )



        # -----------------------------
        # Woman Child
        # -----------------------------

        df["WomanChild"] = (

            (df["Sex"] == "female")

            |

            (df["Age"] < 16)

        ).astype(int)



    # =====================================================
    # Ticket Group Size
    # =====================================================

    ticket_counts = (

        train["Ticket"]
        .value_counts()

    )


    train["TicketGroupSize"] = (

        train["Ticket"]
        .map(ticket_counts)
        .fillna(1)

    )


    test["TicketGroupSize"] = (

        test["Ticket"]
        .map(ticket_counts)
        .fillna(1)

    )



    # =====================================================
    # Age Imputation
    # =====================================================

    train["Age"] = train.groupby(

        ["Title", "Pclass"]

    )["Age"].transform(

        lambda x:
        x.fillna(x.median())

    )


    age_lookup = (

        train.groupby(
            ["Title", "Pclass"]
        )["Age"]
        .median()

    )


    def fill_age(row):

        if pd.isna(row["Age"]):

            key = (

                row["Title"],

                row["Pclass"]

            )


            if key in age_lookup.index:

                return age_lookup[key]


            return train["Age"].median()


        return row["Age"]



    test["Age"] = test.apply(

        fill_age,

        axis=1

    )



    train["Age"] = train["Age"].fillna(

        train["Age"].median()

    )


    test["Age"] = test["Age"].fillna(

        train["Age"].median()

    )



    # =====================================================
    # Fare Processing
    # =====================================================

    train["Fare"] = train["Fare"].fillna(

        train["Fare"].median()

    )


    test["Fare"] = test["Fare"].fillna(

        train["Fare"].median()

    )



    # =====================================================
    # Embarked Processing
    # =====================================================

    train["Embarked"] = train["Embarked"].fillna(

        train["Embarked"].mode()[0]

    )


    test["Embarked"] = test["Embarked"].fillna(

        train["Embarked"].mode()[0]

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