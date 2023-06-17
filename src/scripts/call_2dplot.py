import os

from src.mbn_plot import plot_2d


class Input:
    data_path = os.path.join(os.getcwd(), "High_RepresentativeMatrix.csv")
    output_path = os.path.join(os.getcwd(), "High_RepresentativeMatrix_circle.svg")
    labels_path = os.path.join(os.getcwd(), "labels.csv")


def main():
    plot_2d(Input.data_path, output_path=Input.output_path, labels_path=Input.labels_path)


if __name__ == "__main__":
    main()
