from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def download_titanic_data():
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)

    print("Authenticating with Kaggle...")

    api = KaggleApi()
    api.authenticate()

    print("Downloading Titanic dataset...")

    api.competition_download_files(
        "titanic",
        path=data_folder
    )

    zip_path = data_folder / "titanic.zip"

    print("Extracting files...")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_folder)

    zip_path.unlink()

    print("Done! Dataset saved in data/")


if __name__ == "__main__":
    download_titanic_data()