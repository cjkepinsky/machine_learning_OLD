import umap
from babyplots import Babyplot


def umap_reducer():
    embedding = reducer.fit_transform(scaled_penguin_data)
    embedding.shape


def umap_reducer3d(df, random_state=42, n_components=3):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(df)
    reducer3d = umap.UMAP(random_state=random_state, n_components=n_components)
    embedding3d = reducer3d.fit_transform(df)
    bp = Babyplot()
    bp.add_plot(
        embedding3d.tolist(),
        "pointCloud",
        "categories",
        y.values.tolist(),
        {
            "colorScale": "Set2",
            "showLegend": True,
            "folded": True,
            "foldedEmbedding": embedding.tolist()
        })
    html = bp.save_as_html('./babyplot.html')


print("Done")
