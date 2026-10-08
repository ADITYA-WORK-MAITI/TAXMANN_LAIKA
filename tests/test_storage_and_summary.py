from laika.documents import split_fixed
from laika.storage import ConversationStore, to_csv
from laika.summarizer import summarize
from tests.fakes import ScriptedModel

CONVERSATION = [{"question": "Q1, with comma", "response": "A1"}]


def test_store_round_trip(tmp_path):
    store = ConversationStore(tmp_path / "nested" / "conversations.json")
    assert store.load_all() == {}
    name = store.save(CONVERSATION, "first")
    assert name == "first"
    assert store.load_all() == {"first": CONVERSATION}


def test_csv_export_quotes_fields():
    assert to_csv(CONVERSATION).splitlines() == ["Question,Response", '"Q1, with comma",A1']


def test_split_fixed():
    assert split_fixed("abcdefg", 3) == ["abc", "def", "g"]
    assert split_fixed("", 3) == []


def test_summarize_short_text_uses_one_call():
    model = ScriptedModel("short summary")
    assert summarize("tiny text", model, chunk_size=100) == "short summary"
    assert len(model.prompts) == 1


def test_summarize_long_text_combines_piece_summaries():
    model = ScriptedModel("s1", "s2", "combined")
    assert summarize("a" * 150, model, chunk_size=100) == "combined"
    assert "s1" in model.prompts[-1] and "s2" in model.prompts[-1]
