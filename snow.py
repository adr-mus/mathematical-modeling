# scirpt used in the project "Plowing"
import os, sys
from collections import defaultdict, namedtuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

colors = cm.Set1.colors

Graph = namedtuple("Graph", ["vertices", "edges", "adj_matrix"])


def build_graph(vertices, single_edges, double_edges):
    edges = gather_edges(single_edges, double_edges)
    adj_matrix = create_adj_matrix(vertices, edges)
    return Graph(vertices, edges, adj_matrix)


def gather_edges(single_edges, double_edges):
    edges = set(single_edges)
    edges.update(double_edges)
    edges.update((j, i) for i, j in double_edges)
    return edges


def create_adj_matrix(vertices, edges):
    adj_matrix = np.zeros((30, 30))
    for i, j in edges:
        adj_matrix[i, j] = np.linalg.norm(vertices[i] - vertices[j])
    return 5 * adj_matrix


def show_graph(G, *, directed=False):
    plt.figure(dpi=120)

    for i, j in G.edges:
        x1, y1 = G.vertices[i]
        x2, y2 = G.vertices[j]
        if directed:
            plt.arrow(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                color="k",
                head_length=20,
                head_width=12,
                length_includes_head=True,
                aa=True,
            )
        else:
            plt.plot([x1, x2], [y1, y2], color="k")

    plt.scatter(G.vertices[:, 0], G.vertices[:, 1], c="b", s=20, zorder=3)

    for i, v in enumerate(G.vertices):
        plt.annotate(i, v + 10)

    plt.ylim((100, 700))
    plt.xlim((70, 1200))

    plt.savefig(os.path.join("wyniki", "dzielnica.png"))
    plt.show()


def partition_graph(G, n_clusters, *, directed=False, show=True):
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(G.vertices)

    partition = defaultdict(set)
    for i, j in G.edges:
        p1, p2 = G.vertices[i], G.vertices[j]
        n1, n2 = clustering.predict(np.vstack([p1, p2]))
        if n1 == n2:
            partition[n1].add(i)
            partition[n1].add(j)
        else:
            d1 = np.linalg.norm(p2 - clustering.cluster_centers_[n1])
            d2 = np.linalg.norm(p1 - clustering.cluster_centers_[n2])
            if d1 <= d2:
                partition[n1].add(i)
                partition[n1].add(j)
            else:
                partition[n2].add(i)
                partition[n2].add(j)

    partition = {k: sorted(vs) for k, vs in partition.items()}

    if show:
        show_partition(G, partition, directed=directed)

        plt.scatter(G.vertices[:, 0], G.vertices[:, 1], c="k", s=20, zorder=3)
        plt.scatter(
            clustering.cluster_centers_[:, 0],
            clustering.cluster_centers_[:, 1],
            marker="x",
            c=clustering.predict(clustering.cluster_centers_),
            zorder=3,
            cmap="Set1",
            vmin=0,
            vmax=8,
        )

        plt.savefig(os.path.join("wyniki", str(n_clusters), "podział.png"))

        plt.show()

    return partition


def show_partition(G, partition, *, directed=False):
    plt.figure(dpi=120)

    for i, j in G.edges:
        for k, vs in partition.items():
            if i in vs and j in vs:
                break
        x1, y1 = G.vertices[i]
        x2, y2 = G.vertices[j]
        if directed:
            plt.arrow(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                color=colors[k],
                head_length=20,
                head_width=12,
                length_includes_head=True,
            )
        else:
            plt.plot([x1, x2], [y1, y2], color=colors[k])

    for i, v in enumerate(G.vertices):
        plt.annotate(i, v + 10)

    plt.ylim((100, 700))
    plt.xlim((70, 1200))


def score(G, partition, wa, wc, va, cd, f):
    ts = {}  # potrzebny czas dla pługu k
    ds = {}  # dystans do przebycia dla pługu k
    eps = {}  # ścieżka Eulera dla pługu k
    for k, vs in partition.items():
        sub_adj_matrix = G.adj_matrix[vs][:, vs]

        d = np.sum(sub_adj_matrix)
        t = d / va
        ep = eulerian_path(sub_adj_matrix)
        ts[k] = t
        ds[k] = d
        eps[k] = [vs[i] for i in ep]

    for k in partition:
        print("Podgraf", k)
        print("Ściezka Eulera:")
        print(*eps[k], sep=" -> ")
        print(f"Całkowity dystans do przebycia: {round(ds[k], 2)} m")
        print(f"Potrzebny czas: {round(ts[k], 2)} s\n")

    zr = max(ts.values())  # czas odśnieżania
    zc = sum(ts.values())  # czas użytkowania pługów

    print(f"Całkowity czas odśnieżania: {round(zr, 2)} s")
    print(f"Całkowity czas użytkowania pługów: {round(zc, 2)} s\n")

    objective = wa * zr + wc * (cd * zc + f * len(partition))

    print(f"Funkcja celu: {round(objective, 2)}")

    return objective


def eulerian_path(adj_matrix):
    # algorytm Hierholzera
    edges = {i: list(adj_matrix[i].nonzero()[0]) for i in range(len(adj_matrix))}

    curr_path = [0]
    circuit = []
    curr_v = 0
    while len(curr_path) != 0:
        if len(edges[curr_v]) != 0:
            curr_path.append(curr_v)
            curr_v = edges[curr_v].pop()
        else:
            circuit.append(curr_v)
            curr_v = curr_path[-1]
            curr_path.pop()
    circuit.reverse()

    return circuit


def main(G, n_clusters, *, va, cd, f, wa, wc, directed):
    if str(n_clusters) not in os.listdir("wyn"):
        os.mkdir(os.path.join("wyn", str(n_clusters)))
        sys.stdout = open(
            os.path.join("wyn", str(n_clusters), "wynik.txt"), "w", encoding="UTF-8"
        )

        partition = partition_graph(G, n_clusters, directed=True)
        return score(G, partition, wa, wc, va, cd, f)


if __name__ == "__main__":
    vertices = np.array(
        [
            (119, 410),
            (233, 417),
            (237, 259),
            (357, 269),
            (353, 417),
            (358, 544),
            (494, 409),
            (633, 311),
            (510, 218),
            (782, 320),
            (798, 204),
            (711, 194),
            (791, 150),
            (930, 198),
            (930, 149),
            (1061, 187),
            (935, 315),
            (1083, 312),
            (1118, 439),
            (1084, 491),
            (949, 457),
            (789, 459),
            (633, 466),
            (695, 609),
            (551, 621),
            (950, 626),
            (947, 556),
            (944, 499),
            (1055, 543),
            (1035, 617),
        ]
    )
    double_edges = [
        (0, 1),
        (1, 2),
        (1, 4),
        (4, 3),
        (4, 5),
        (4, 6),
        (6, 7),
        (7, 8),
        (7, 9),
        (9, 10),
        (9, 16),
        (10, 11),
        (10, 12),
        (10, 13),
        (13, 14),
        (13, 15),
        (15, 17),
        (13, 16),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 28),
        (28, 29),
        (29, 25),
        (25, 26),
        (26, 28),
        (26, 27),
        (27, 19),
        (27, 20),
        (20, 18),
        (20, 21),
        (21, 22),
        (22, 6),
        (22, 24),
        (22, 23),
        (23, 25),
    ]
    single_edges = []

    G = build_graph(vertices, single_edges, double_edges)
    show_graph(G, directed=True)

    n_clusters = 3
    va, cd, f = 4, 1, 400
    wa, wc = 1, 2
    directed = True

    main(G, n_clusters, va=va, cd=cd, f=f, wa=wa, wc=wc, directed=directed)
