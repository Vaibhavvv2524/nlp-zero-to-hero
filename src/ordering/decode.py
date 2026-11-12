"""Turn pairwise mBERT scores into a sentence order.

Pipeline
--------
1. Cluster sentences with TF-IDF (char 2-4 grams) + KMeans. Clustering keeps
   the number of expensive pairwise model calls small and follows the notes.
2. Score cluster medoids with the mBERT pair classifier to order the clusters.
3. Score all sentence pairs inside each cluster and find the best chain with
   beam search.
4. Concatenate the cluster orders.

Kendall's tau is used to measure how well a shuffled held-out article is
recovered.
"""

import random
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..constants import (
    BASE_MODEL,
    MEMORY_LIMIT_GB,
    ORDER_MAX_LENGTH,
    ORDER_MODEL_DIR,
    SEED,
)
from ..utils import (
    free_memory,
    get_device,
    gpu_memory,
    limit_gpu_memory,
    read_jsonl,
)


class OrderScorer:
    """Load the LoRA pair classifier and expose symmetric before/after scores."""

    def __init__(
        self,
        model_dir=ORDER_MODEL_DIR,
        device=None,
        memory_gb=MEMORY_LIMIT_GB,
    ):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.torch = torch
        self.memory_report = limit_gpu_memory(memory_gb)
        self.device = device or get_device()

        adapter_dir = Path(model_dir) / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"No ordering adapter found at {adapter_dir}. "
                "Train it first: python -m src.ordering.train"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=2,
            id2label={0: "wrong_order", 1: "correct_order"},
            label2id={"wrong_order": 0, "correct_order": 1},
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

    def _pair_probabilities(self, first, second, batch_size=32):
        """Return ``P(first comes before second)`` for each pair."""
        import torch

        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(first), batch_size):
                batch_first = first[start : start + batch_size]
                batch_second = second[start : start + batch_size]
                encoding = self.tokenizer(
                    batch_first,
                    batch_second,
                    truncation=True,
                    max_length=ORDER_MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                )
                encoding = {
                    key: value.to(self.device)
                    for key, value in encoding.items()
                }
                logits = self.model(**encoding).logits.float()
                probabilities.extend(
                    torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                )
        return probabilities

    def score_matrix(self, sentences, batch_size=32):
        """Symmetric matrix where ``M[i, j] = P(sentence i before j)``."""
        count = len(sentences)
        # The diagonal compares a sentence with itself, so 0.5 is neutral.
        matrix = np.full((count, count), 0.5, dtype=np.float64)
        # Score each unordered pair once, then fill both sides with
        # P(i before j) and 1 - P(i before j).
        pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
        if not pairs:
            return matrix
        first = [sentences[i] for i, _ in pairs]
        second = [sentences[j] for _, j in pairs]
        probabilities = self._pair_probabilities(first, second, batch_size)
        for (i, j), probability in zip(pairs, probabilities):
            matrix[i, j] = probability
            matrix[j, i] = 1.0 - probability
        return matrix

    def predict_pair_labels(self, pairs, batch_size=32):
        """Return 1 when A is predicted before B, otherwise 0."""
        first = [pair[0] for pair in pairs]
        second = [pair[1] for pair in pairs]
        probabilities = self._pair_probabilities(first, second, batch_size)
        return [1 if probability >= 0.5 else 0 for probability in probabilities]

    def close(self):
        del self.model
        self.model = None
        free_memory()

    def memory(self):
        return gpu_memory()


# ---------------------------------------------------------------------------
# Clustering and ordering
# ---------------------------------------------------------------------------
def cluster_sentences(sentences, n_clusters=None):
    """Return ``(clusters, medoids)`` using TF-IDF + KMeans."""
    count = len(sentences)
    if count < 4:
        return [list(range(count))], [0]

    if n_clusters is None:
        n_clusters = int(round(count / 6))
    n_clusters = max(1, min(12, n_clusters, count))

    if n_clusters == 1:
        return [list(range(count))], [0]

    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
    ).fit_transform(sentences)
    labels = KMeans(
        n_clusters=n_clusters, n_init="auto", random_state=SEED
    ).fit_predict(matrix)

    clusters = []
    medoids = []
    for cluster_id in range(n_clusters):
        members = np.where(labels == cluster_id)[0]
        if len(members) == 0:
            continue
        centroid = np.asarray(matrix[members].mean(axis=0))
        similarity = cosine_similarity(matrix[members], centroid).ravel()
        medoid = int(members[int(np.argmax(similarity))])
        clusters.append(members.tolist())
        medoids.append(medoid)
    return clusters, medoids


def _earliness(matrix, node, nodes):
    """Average probability that ``node`` comes before the other nodes."""
    others = [other for other in nodes if other != node]
    if not others:
        return 0.0
    return float(np.mean([matrix[node, other] for other in others]))


def _greedy_order(matrix, nodes):
    """Start with the earliest sentence, then repeatedly take the next best."""
    nodes = list(nodes)
    if len(nodes) <= 1:
        return nodes
    current = max(nodes, key=lambda node: _earliness(matrix, node, nodes))
    order = [current]
    remaining = set(nodes)
    remaining.remove(current)
    while remaining:
        nxt = max(remaining, key=lambda node: matrix[current, node])
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def _beam_order(matrix, nodes, beam_width=8):
    """Beam search over chains, scoring each adjacent pair with ``matrix``."""
    nodes = list(nodes)
    if len(nodes) <= 1:
        return nodes

    starts = sorted(
        nodes, key=lambda node: _earliness(matrix, node, nodes), reverse=True
    )[:beam_width]
    beams = [([node], set(nodes) - {node}, 0.0) for node in starts]

    while beams and len(beams[0][0]) < len(nodes):
        candidates = []
        for path, remaining, score in beams:
            for nxt in remaining:
                candidates.append(
                    (
                        path + [nxt],
                        remaining - {nxt},
                        score + matrix[path[-1], nxt],
                    )
                )
        candidates.sort(key=lambda item: item[2], reverse=True)
        beams = candidates[:beam_width]

    return beams[0][0]


STRATEGIES = ("borda", "clusters", "beam")


def order_from_matrix(
    matrix,
    sentences,
    strategy="borda",
    n_clusters=None,
    beam_width=8,
):
    """Decode one strategy from a precomputed pairwise score matrix."""
    count = len(sentences)
    if count <= 1:
        return list(range(count))

    if strategy == "borda":
        # Borda/Kemeny approximation: a sentence that is often "before" the
        # others gets a high average score and moves to the front. Averaging
        # all pairs is more robust than chaining only adjacent decisions.
        nodes = list(range(count))
        return sorted(
            nodes, key=lambda node: _earliness(matrix, node, nodes), reverse=True
        )

    if strategy == "beam":
        return _beam_order(matrix, range(count), beam_width)

    if strategy == "clusters":
        clusters, medoids = cluster_sentences(sentences, n_clusters)
        if len(clusters) == 1:
            return _beam_order(matrix, range(count), beam_width)

        medoid_matrix = matrix[np.ix_(medoids, medoids)]
        cluster_order = _greedy_order(medoid_matrix, range(len(medoids)))
        ordered = []
        for cluster_index in cluster_order:
            members = clusters[cluster_index]
            submatrix = matrix[np.ix_(members, members)]
            local_order = _beam_order(
                submatrix, range(len(members)), beam_width
            )
            ordered.extend(members[position] for position in local_order)
        return ordered

    raise ValueError(
        f"Unknown strategy {strategy!r}; choose from {STRATEGIES}"
    )


def order_sentences(
    sentences,
    scorer,
    strategy="borda",
    n_clusters=None,
    beam_width=8,
):
    """Return an ordering of ``sentences`` as a list of original indices."""
    count = len(sentences)
    if count <= 1:
        return list(range(count))
    matrix = scorer.score_matrix(sentences)
    return order_from_matrix(
        matrix, sentences, strategy, n_clusters, beam_width
    )


def order_sentences_tfidf(sentences):
    """Ablation: order sentences using only TF-IDF similarity."""
    count = len(sentences)
    if count <= 1:
        return list(range(count))
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True
    ).fit_transform(sentences)
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)
    return _greedy_order(similarity, range(count))


def kendall_tau(recovered, truth):
    """Compare a recovered order with the true order (-1 to 1)."""
    if len(recovered) < 2:
        return float("nan")
    return float(kendalltau(recovered, truth).statistic)


def evaluate_orderings(
    eval_path,
    scorer,
    max_articles=40,
    seed=SEED,
    strategies=STRATEGIES,
):
    """Shuffle held-out articles, recover the order and report Kendall's tau.

    One pairwise matrix is computed per article and reused by every strategy,
    so comparing the decoders is cheap.
    """
    rows = read_jsonl(eval_path)[:max_articles]
    rng = random.Random(seed)
    scores = {strategy: [] for strategy in strategies}
    scores["tfidf_ablation"] = []

    for row in rows:
        sentences = row["sentences"]
        count = len(sentences)
        if count < 4:
            continue
        truth = list(range(count))
        shuffled_indices = truth.copy()
        rng.shuffle(shuffled_indices)
        shuffled = [sentences[index] for index in shuffled_indices]

        matrix = scorer.score_matrix(shuffled)
        for strategy in strategies:
            predicted = order_from_matrix(matrix, shuffled, strategy)
            recovered = [shuffled_indices[index] for index in predicted]
            scores[strategy].append(kendall_tau(recovered, truth))

        tfidf_order = order_sentences_tfidf(shuffled)
        tfidf_recovered = [shuffled_indices[index] for index in tfidf_order]
        scores["tfidf_ablation"].append(kendall_tau(tfidf_recovered, truth))

    def summarize(values):
        clean = [value for value in values if not np.isnan(value)]
        if not clean:
            return {"articles": 0, "mean_tau": None, "median_tau": None}
        return {
            "articles": len(clean),
            "mean_tau": round(float(np.mean(clean)), 4),
            "median_tau": round(float(np.median(clean)), 4),
            "min_tau": round(float(np.min(clean)), 4),
            "max_tau": round(float(np.max(clean)), 4),
        }

    return {name: summarize(values) for name, values in scores.items()}
