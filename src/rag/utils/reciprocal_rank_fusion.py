from collections import defaultdict

from rag.models.minimal_source import MinimalSource


def reciprocal_rank_fusion(
    rankings: list[list[MinimalSource]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[MinimalSource]:
    if weights is None:
        weights = [1.0] * len(rankings)
    rrf_scores: dict[MinimalSource, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, document in enumerate(ranking, start=1):
            rrf_scores[document] += weight * (1 / (k + rank - 1))
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
