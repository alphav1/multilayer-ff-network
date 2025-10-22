import kagglehub
from kagglehub import KaggleDatasetAdapter


def dataset():
    file_path = r"D:\University\Neural Networks\multilayer-ff-network\data\raw"

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "oddrationale/mnist-in-csv",
        file_path,
    )

    print("First 5 records:", df.head())
    return df


if __name__ == "__main__":
    dataset()
