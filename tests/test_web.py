import json
import pytest
from fastapi.testclient import TestClient
from src import conversation_store as store
from api import app as api
from src.speech import ProviderError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'DB_PATH', str(tmp_path / 'conversations.db'))
    with TestClient(api.app) as client:
        yield client


def events(response):
    return [json.loads(frame[6:]) for frame in response.text.strip().split('\n\n')]


def test_web_and_input_validation(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    assert client.post('/chat', json={'question': '   '}).status_code == 422
    assert client.post('/chat', json={'question': 'test', 'language': 'unknown'}).status_code == 422
    assert client.post('/chat', json={'question': 'test', 'conversation_id': 'missing'}).status_code == 404
    assert client.post('/voice', files={'file': ('bad.txt', b'text', 'text/plain')}).status_code == 415
    assert client.post('/voice', files={'file': ('empty.webm', b'', 'audio/webm')}).status_code == 413


def test_chat_history_and_delete(client, monkeypatch):
    calls = []
    def graph(question, mode, language, history):
        calls.append((question, history))
        return {'answer': 'Bring your supporting documents.', 'relevant_docs': []}
    monkeypatch.setattr(api, 'run_graph', graph)
    first = events(client.post('/chat', json={'question': 'Renew CNIC?'}))
    conv_id = first[0]['id']
    assert any(e.get('text') == 'Bring your supporting documents.' for e in first)
    assert first[-1]['type'] == 'done'
    events(client.post('/chat', json={'question': 'What else?', 'conversation_id': conv_id}))
    assert calls[1][1][0] == {'role': 'user', 'content': 'Renew CNIC?'}
    assert len(client.get('/conversations/' + conv_id).json()['messages']) == 4
    assert client.get('/conversations').json()['conversations'][0]['message_count'] == 4
    assert client.delete('/conversations/' + conv_id).status_code == 200
    assert client.get('/conversations/' + conv_id).status_code == 404
    assert client.get('/conversations').json()['conversations'] == []


def test_voice_keeps_answer_when_speech_fails(client, monkeypatch):
    async def transcribe(*args): return 'My spoken question'
    async def translate(text, lang): return text
    async def speech(*args): raise ProviderError('Uplift AI quota or rate limit reached.')
    monkeypatch.setattr(api, 'transcribe', transcribe)
    monkeypatch.setattr(api, 'translate_to_english', translate)
    monkeypatch.setattr(api, 'synthesize', speech)
    monkeypatch.setattr(api, 'run_graph', lambda *a: {'answer': 'The answer', 'relevant_docs': []})
    result = events(client.post('/voice', files={'file': ('voice.webm', b'audio', 'audio/webm')}, data={'language': 'urdu'}))
    assert any(e['type'] == 'transcript' and e['text'] == 'My spoken question' for e in result)
    assert any(e['type'] == 'answer' and e['text'] == 'The answer' for e in result)
    assert any(e['type'] == 'error' and 'quota' in e['text'] for e in result)
    assert result[-1]['type'] == 'done'
    assert len(client.get('/conversations/' + result[0]['id']).json()['messages']) == 2


def test_provider_failure_is_not_saved_as_answer(client, monkeypatch):
    def graph(*args): raise RuntimeError('secret provider internals')
    monkeypatch.setattr(api, 'run_graph', graph)
    result = events(client.post('/chat', json={'question': 'Hello'}))
    assert any(e['type'] == 'error' for e in result)
    assert 'secret provider internals' not in str(result)
    assert result[-1]['type'] == 'done'
    assert client.get('/conversations/' + result[0]['id']).json()['messages'] == []


def test_local_retriever_finds_bundled_nadra_context():
    from src.local_retrieval import LocalRetriever
    retriever = LocalRetriever()
    docs = retriever.invoke('CNIC renewal documents')
    assert docs and all(doc.metadata['source'].endswith('.txt') for doc in docs)
    assert any('CNIC' in doc.page_content for doc in docs)
    renewal = retriever.invoke('What documents are required to renew a CNIC?')
    assert any('Scanned image of ID card' in doc.page_content for doc in renewal)
    assert retriever.invoke('') == []


def test_gemini_quota_is_explained(client, monkeypatch):
    class GoogleRateLimitError(Exception): pass
    def graph(*args):
        try:
            raise GoogleRateLimitError('private provider details')
        except GoogleRateLimitError as exc:
            raise RuntimeError('verification failed') from exc
    monkeypatch.setattr(api, 'run_graph', graph)
    result = events(client.post('/chat', json={'question': 'Hello'}))
    assert any(e['type'] == 'error' and 'quota' in e['text'] for e in result)
    assert 'private provider details' not in str(result)


def test_relevance_is_batched_and_preserves_order(monkeypatch):
    from src import nodes
    from langchain_core.documents import Document
    from src.models import BatchRelevanceDecision
    calls = []
    class Grader:
        def invoke(self, messages):
            calls.append(messages)
            return BatchRelevanceDecision(relevant_indices=[2, 0, 2, 99])
    monkeypatch.setattr(nodes, 'batch_relevance_llm', Grader())
    docs = [Document(page_content=text) for text in ['renewal', 'unrelated', 'CNIC']]
    assert nodes.is_relevant({'question': 'renew CNIC', 'docs': docs})['relevant_docs'] == [docs[0], docs[2]]
    assert len(calls) == 1


def test_gemini_budget_waits_after_limit(monkeypatch):
    from src import rate_limit
    clock = [0.0]
    monkeypatch.setattr(rate_limit.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(rate_limit.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    limiter = rate_limit.GeminiRateLimiter()
    limiter.limit = 2
    assert limiter.acquire()
    assert limiter.acquire()
    assert not limiter.acquire(blocking=False)
    assert limiter.acquire()
    assert clock[0] >= 61


def test_local_storage_ignores_cloud_credentials(monkeypatch):
    import runpy
    from pathlib import Path
    monkeypatch.setenv('STORAGE_MODE', 'local')
    monkeypatch.setenv('SUPABASE_DB_URL', 'postgresql://unused.invalid/db')
    monkeypatch.setenv('QDRANT_URL', 'https://unused.invalid')
    monkeypatch.setenv('QDRANT_API_KEY', 'unused')
    config = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/config.py'))
    assert config['SUPABASE_DB_URL'] is None
    assert config['QDRANT_URL'] is None
    assert config['QDRANT_API_KEY'] is None


def test_local_ingestion_never_opens_cloud_connections(tmp_path, monkeypatch):
    from src import ingestion
    monkeypatch.setattr(ingestion, 'STORAGE_MODE', 'local')
    def forbid_cloud():
        raise AssertionError('Local ingestion must not connect to PostgreSQL')
    monkeypatch.setattr(ingestion, '_get_pg_pool', forbid_cloud)
    (tmp_path / 'test.txt').write_text('CNIC renewal guidance')
    ingestion.ingest_all(str(tmp_path))
