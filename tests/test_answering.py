from laika import answering, prompts
from laika.answering import answer_question, canned_answer, find_similar_turn
from laika.retrieval import HybridIndex
from tests.fakes import HashEmbeddings, ScriptedModel

EMBED = HashEmbeddings().embed_query


def test_canned_answers():
    assert canned_answer("Who are you?", 0).startswith("I am Laika")
    assert canned_answer("How many PDFs have I uploaded?", 3) == "You have uploaded 3 document(s)."
    assert canned_answer("What is GST?", 0) is None


def test_find_similar_turn():
    history = [
        {"question": "what is the due date for income tax returns", "response": "31 July"},
        {"question": "where does it rain most", "response": "Kerala"},
    ]
    assert find_similar_turn("what is the due date for income tax returns?", history, EMBED)["response"] == "31 July"
    assert find_similar_turn("completely unrelated words here", history, EMBED) is None
    assert find_similar_turn("anything", [], EMBED) is None


def test_answers_from_documents_when_possible():
    index = HybridIndex(["The GST rate on books is zero."], HashEmbeddings())
    model = ScriptedModel("gst on books", "Books are taxed at **0%**.")
    answer = answer_question("What is the GST rate on books?", index, [], model, EMBED, 1)
    assert answer == "Books are taxed at **0%**."
    assert "The GST rate on books is zero." in model.prompts[-1]


def test_falls_back_to_web_when_documents_lack_the_answer(monkeypatch):
    index = HybridIndex(["The GST rate on books is zero."], HashEmbeddings())
    model = ScriptedModel("", prompts.NOT_FOUND, "Paris is the capital of France.")
    monkeypatch.setattr(answering, "DDGS", lambda: _FakeSearch([{"title": "t", "href": "u", "body": "b"}]))
    answer = answer_question("What is the capital of France?", index, [], model, EMBED, 1)
    assert "on the web" in answer
    assert answer.endswith("Paris is the capital of France.")


def test_falls_back_to_wikipedia_when_web_search_fails(monkeypatch):
    monkeypatch.setattr(answering, "DDGS", lambda: _FakeSearch(error=True))
    monkeypatch.setattr(answering.wikipedia, "summary", lambda *args, **kwargs: "Wiki text.")
    answer = answer_question("Capital of France?", None, [], ScriptedModel(), EMBED)
    assert answer.endswith("Wiki text.")


class _FakeSearch:
    def __init__(self, results=None, error=False):
        self.results, self.error = results, error

    def text(self, *args, **kwargs):
        if self.error:
            raise RuntimeError("network down")
        return self.results
