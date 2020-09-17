import math
import random
import time
from collections import defaultdict
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sum_sets(sets):
    ret = set()
    for s in sets:
        ret.update(s)
    return ret


def plot_simulation(n, m, *, save=False):
    # przygotowanie siatki
    population = np.arange(n)
    generations = np.arange(m + 1)
    xx, yy = np.meshgrid(population, generations)
    xs, ys = np.ravel(xx), np.ravel(yy)

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c="k")
    plt.axis("scaled")
    plt.xticks(population, labels=population + 1)
    plt.yticks(generations)
    plt.xlabel("Osoba")
    plt.ylabel("Pokolenie wstecz")

    related = [{j} for j in population]
    # related[j] = zbiór osób z pokolenia 0 spokrewnionych z j-tą osobą z obecnego pokolenia
    ucas = set()
    # (j, i) \in ucas => j-ta osoba w i-tym pokoleniu jest WP
    exts = set()
    # (j, i) \in exts => j-ta osoba w i-tym pokoleniu nie ma żyjących potomków
    for i in range(1, m + 1):  # symulacja - część zasadnicza
        children = defaultdict(list)
        for j in population:
            for _ in range(2):  # losowanie rodziców
                parent = random.randrange(n)
                children[parent].append(j)
                plt.plot([j, parent], [i - 1, i], color="k", linewidth=1)

        related = [sum_sets(related[k] for k in children[j]) for j in population]
        for j, rel in enumerate(related):
            if len(rel) == n:
                ucas.add((j, i))
            elif len(rel) == 0:
                exts.add((j, i))

    plt.scatter(*zip(*ucas), c="b", zorder=3)
    plt.scatter(*zip(*exts), c="r", zorder=3)

    if save:
        plt.savefig(r"symulacja.jpg")

    plt.show()


def run_simulation(n):
    related = [{j} for j in range(n)]
    Tn = None
    for i in count(1):
        children = defaultdict(list)
        for j in range(n):
            for _ in range(2):
                children[random.randrange(n)].append(j)

        related = [sum_sets(related[k] for k in children[j]) for j in range(n)]
        if Tn is None and any(len(rel) == n for rel in related):
            Tn = i
        if all(len(rel) in (0, n) for rel in related):
            Un = i
            break

    return Tn, Un


if __name__ == "__main__":
    # ilustruje działanie modelu
    n = 6  # liczebność populacji
    m = 5  # liczba pokoleń wstecz

    plot_simulation(n, m, save=False)
    ##########################################################
    # symulacja Tn i Un
    ns = [50, 100, 250, 500]  # liczebności populacji
    N = 50  # liczba symulacji na populację

    st = time.perf_counter()
    rows = []
    for n in ns:
        Tn, Un = np.mean([run_simulation(n) for _ in range(N)], axis=0)
        rows.append((n, Tn, Tn / math.log2(n), Un, Un / (1.7689 * math.log2(n))))

    col_names = ["n", "Tn", "Tn/log2(n)", "Un", "Un/((1 + C)log2(n))"]
    summary = pd.DataFrame(dict(zip(col_names, zip(*rows)))).round(4)
    summary.set_index("n", inplace=True)

    # with open(r"podsum_tex.txt", "w") as f:
    #     summary.to_latex(f)

    print(summary)
    print("Zakończono w ok.", round(time.perf_counter() - st, 2), "s.")
