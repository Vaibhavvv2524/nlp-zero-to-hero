"""Sentence-ordering baseline: TF-IDF similarity + clustering + greedy paths.

Runs on CPU. Kendall's tau is measured on a synthetic chain whose adjacent
sentences share a unique bridge word.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .segment import END_WORDS, STARTERS
except ImportError:
    from segment import END_WORDS, STARTERS


def _vectorize(sentences):
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
    )
    return vectorizer.fit_transform(sentences)


def _best_greedy_path(similarity, nodes):
    nodes = list(nodes)
    if len(nodes) <= 2:
        return nodes

    best_path = nodes
    best_score = float("-inf")

    for start in nodes:
        path = [start]
        remaining = set(nodes)
        remaining.remove(start)
        score = 0.0
        while remaining:
            current = path[-1]
            nxt = max(remaining, key=lambda index: similarity[current, index])
            score += similarity[current, nxt]
            path.append(nxt)
            remaining.remove(nxt)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _orientation_score(path, sentences):
    first_words = sentences[path[0]].split()
    last_words = sentences[path[-1]].split()
    score = 0.0
    if first_words and first_words[0] in STARTERS:
        score += 2.0
    if last_words and last_words[-1] in END_WORDS:
        score += 1.0
    return score


def _orient(path, sentences):
    reversed_path = list(reversed(path))
    if _orientation_score(reversed_path, sentences) > _orientation_score(
        path, sentences
    ):
        return reversed_path
    return path


def _order_clusters(centroid_similarity):
    count = centroid_similarity.shape[0]
    start = int(np.argmax(centroid_similarity.sum(axis=1)))
    order = [start]
    remaining = set(range(count))
    remaining.remove(start)
    while remaining:
        current = order[-1]
        nxt = max(remaining, key=lambda c: centroid_similarity[current, c])
        order.append(nxt)
        remaining.remove(nxt)
    return order


def order_sentences(sentences, n_clusters=None):
    """Return indices that place ``sentences`` into a coherent order."""
    count = len(sentences)
    if count <= 2:
        return _orient(list(range(count)), sentences)

    matrix = _vectorize(sentences)
    similarity = cosine_similarity(matrix)

    if n_clusters is None:
        n_clusters = max(1, min(6, count // 12))
    n_clusters = max(1, min(n_clusters, count))

    if n_clusters == 1:
        return _orient(_best_greedy_path(similarity, range(count)), sentences)

    labels = KMeans(
        n_clusters=n_clusters, n_init="auto", random_state=42
    ).fit_predict(matrix)
    clusters = [np.where(labels == cluster)[0] for cluster in range(n_clusters)]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]

    centroids = np.vstack(
        [np.asarray(matrix[cluster].mean(axis=0)) for cluster in clusters]
    )
    cluster_order = _order_clusters(cosine_similarity(centroids))

    ordered = []
    for cluster_index in cluster_order:
        ordered.extend(_best_greedy_path(similarity, clusters[cluster_index]))
    return _orient(ordered, sentences)


def kendall_tau(recovered, truth):
    from scipy.stats import kendalltau

    return float(kendalltau(recovered, truth).statistic)


def synthetic_chain(n=24, seed=0):
    rng = np.random.default_rng(seed)
    shared = ["विषय", "वाक्य", "पाठ", "कथा", "विवरण"]
    sentences = []
    for index in range(n):
        words = list(shared) + [f"कड़ी{index}", f"कड़ी{index + 1}"]
        rng.shuffle(words)
        if index == 0:
            words.insert(0, "पहले")
        if index == n - 1:
            words.append("था")
        sentences.append(" ".join(words))
    truth = list(range(n))
    shuffled = truth.copy()
    rng.shuffle(shuffled)
    return [sentences[index] for index in shuffled], shuffled


def evaluate_synthetic(n=24, seed=0):
    sentences, truth = synthetic_chain(n=n, seed=seed)
    ordered = order_sentences(sentences, n_clusters=1)
    recovered = [truth[index] for index in ordered]
    return kendall_tau(recovered, list(range(n)))


if __name__ == "__main__":
    for size in (12, 24, 48):
        print(f"synthetic n={size} kendall_tau={evaluate_synthetic(n=size):.3f}")
