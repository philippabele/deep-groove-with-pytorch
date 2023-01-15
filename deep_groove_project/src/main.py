import torch
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("../data/first_dataset.csv")
    show_standard_info(df)
    show_correlation_grafs(df)
    show_3d_scatter(df)


def show_standard_info(df):
    print("\n General Info:")
    df.info()
    print("\n Table:")
    print(df.head())


def show_correlation_grafs(df):
    fields = ['Fr','n','Lifetime']

    chart = alt.Chart(df).mark_point().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative')
    ).properties(
        width=200,
        height=200
    ).repeat(
        row=fields,
        column=fields[::1]
    ).interactive()
    chart.show()


def show_3d_scatter(df):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    fr = df.iloc[:, 0]
    n = df.iloc[:, 1]
    lifetime = df.iloc[:, 2]

    ax.scatter(fr, n, lifetime)
    ax.set_xlabel('Fr')
    ax.set_ylabel('n')
    ax.set_zlabel('Lifetime')
    plt.show()


if __name__ == "__main__":
    main()
