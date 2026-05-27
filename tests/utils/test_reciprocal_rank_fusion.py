from rag.utils.reciprocal_rank_fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    rankings = [["3", "1", "2", "4"], ["2", "3", "1"]]
    rrf_ranks = reciprocal_rank_fusion(rankings)
    assert rrf_ranks == ["3", "2", "1", "4"]

    rankings = [["1", "5", "2", "4", "3"], ["4", "3", "1", "2"]]
    rrf_ranks = reciprocal_rank_fusion(rankings)
    assert rrf_ranks == ["1", "4", "3", "2", "5"]
