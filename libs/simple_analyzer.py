from libs.simpleplotter import simple_features_overview, simple_correlations
from libs.simpleplotter import simple_heatmap


def print_analytics(df, target_col=None):
    print("Data Overview:")
    print(df.head())

    print()
    print("Data Info:")
    print(df.info())

    print()
    print("Data Info2:")
    print(df.describe())

    print()
    print("Missing data:")
    print(df.isnull().sum())

    print()
    print("Features:")
    simple_features_overview(df)

    print()
    print("Correlations:")
    if target_col is not None:
        simple_correlations(df, target_col)
    simple_heatmap(df)
    print_simple_correlations(df)


def print_simple_correlations(df, level=0.3):
    print("Correlations >= " + str(level) + ":")
    for a in range(len(df.corr().columns)):
        col_a = df.corr().columns[a]
        print()
        print(col_a + ":")
        for b in range(a):
            corr = df.corr().iloc[a, b]
            if abs(corr) >= level:
                col_b = df.corr().columns[b]
                print(" - " + col_b + ": {:.2f}" . format(corr))
    print("Done")

