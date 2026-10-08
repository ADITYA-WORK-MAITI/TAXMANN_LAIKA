import pytest

from laika.retrieval import (
    HybridIndex,
    cosine_similarity,
    generate_query_variants,
    parse_query_variants,
    reciprocal_rank_fusion,
)
from tests.fakes import HashEmbeddings, ScriptedModel

CHUNKS = [
    "Goods and services tax is charged on the supply of goods.",
    "Income tax returns must be filed before the due date in July.",
    "Section 80C allows a deduction for investments in provident funds.",
    "The monsoon season brings heavy rain to Kerala.",
]


def test_rrf_prefers_items_ranked_high_in_many_lists():
    assert reciprocal_rank_fusion([[1, 2, 3], [2, 1], [2]]) == [2, 1, 3]


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_parse_query_variants_strips_numbering_and_blank_lines():
    reply = "1. first query\n\n- second query\n3) third query\nfourth query"
    assert parse_query_variants(reply, 3) == ["first query", "second query", "third query"]


def test_generate_query_variants_keeps_original_question_first():
    model = ScriptedModel("what is gst\ngst meaning")
    assert generate_query_variants("Explain GST", model) == ["Explain GST", "what is gst", "gst meaning"]


def test_generate_query_variants_survives_model_errors():
    def broken(_prompt):
        raise RuntimeError("quota exceeded")

    assert generate_query_variants("Explain GST", broken) == ["Explain GST"]


def test_hybrid_index_finds_relevant_chunk():
    index = HybridIndex(CHUNKS, HashEmbeddings())
    results = index.search(["deduction for provident fund investments"], top_k=2)
    assert results[0] == CHUNKS[2]


def test_keyword_ranking_ignores_chunks_without_matching_words():
    index = HybridIndex(CHUNKS, HashEmbeddings())
    assert index.keyword_ranking("monsoon", top_k=4) == [3]


def test_hybrid_index_rejects_empty_text():
    with pytest.raises(ValueError):
        HybridIndex([], HashEmbeddings())
