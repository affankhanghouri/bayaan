"""Tafseel browser app and Flutter-compatible SSE API. Run: uvicorn api.app:app"""
import asyncio
import base64
import json
import logging
import os
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.graph import app as nia_graph
from src import conversation_store as store
from src.config import STORAGE_MODE, QDRANT_URL
from src.speech import ProviderError, transcribe, translate_to_english, synthesize

logger = logging.getLogger(__name__)
WEB = Path(__file__).resolve().parent.parent / 'web'
Language = Literal['english', 'urdu', 'sindhi', 'balochi']
MAX_AUDIO = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(store.init_conversation_tables)
    yield


app = FastAPI(title='Tafseel API', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=WEB), name='static')


@app.get('/', include_in_schema=False)
async def index():
    return FileResponse(WEB / 'index.html')


def event(kind, **data):
    return 'data: ' + json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n\n'


def stream(generator):
    return StreamingResponse(generator, media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


async def resolve(conversation_id, mode):
    if conversation_id:
        conv = await asyncio.to_thread(store.get_full_conversation, conversation_id)
        if not conv:
            raise HTTPException(404, 'Conversation not found')
        return conversation_id, await asyncio.to_thread(store.get_conversation_history, conversation_id)
    return await asyncio.to_thread(store.create_conversation, mode), []


def run_graph(question, mode, language, history):
    return nia_graph.invoke({
        'question': question, 'mode': mode, 'language': language, 'history': history,
        'conversation_id': None, 'need_retrieval': False, 'docs': [], 'relevant_docs': [],
        'context': '', 'answer': '', 'retrieval_query': '', 'retries': 0, 'rewrite_tries': 0,
        'issup': '', 'evidence': [], 'is_useful': False,
    }, config={'recursion_limit': 150})


async def answer_events(question, conv_id, language, history, mode):
    yield event('status', text='Understanding your question…')
    retrieval_question = await translate_to_english(question, language)
    yield event('status', text='Searching and checking the answer…')
    graph_task = asyncio.create_task(asyncio.to_thread(run_graph, retrieval_question, mode, language, history))
    while not graph_task.done():
        done, _ = await asyncio.wait({graph_task}, timeout=15)
        if not done:
            yield event('status', text='Still checking… API rate limits can make this take a little longer.')
    result = await graph_task
    answer = result.get('answer') or 'I could not find an answer in the available documents.'
    await asyncio.to_thread(store.save_turn, conv_id, question, answer, not history)
    sources = sorted({Path(d.metadata.get('source', '')).name for d in result.get('relevant_docs', [])})
    yield event('sources', sources=sources)
    if mode == 'voice':
        yield event('answer', text=answer)
        yield event('status', text='Preparing spoken answer…')
        try:
            audio, mime = await synthesize(answer, language)
            yield event('audio', data=base64.b64encode(audio).decode(), mime_type=mime)
        except (ProviderError, httpx.HTTPError) as exc:
            yield event('error', text=str(exc) if isinstance(exc, ProviderError) else 'Speech service unavailable. Your answer is shown above.')
    else:
        # The graph validates the complete answer before it is delivered.
        yield event('token', text=answer)


async def pipeline(conv_id, language, history, question=None, audio=None, mime=None):
    yield event('conversation_id', id=conv_id)
    try:
        if audio is not None:
            yield event('status', text='Transcribing your recording…')
            question = await transcribe(audio, mime, language)
            yield event('transcript', text=question)
        async for item in answer_events(question, conv_id, language, history, 'voice' if audio is not None else 'text'):
            yield item
    except ProviderError as exc:
        yield event('error', text=str(exc))
    except Exception as exc:
        logger.error('Request failed: %s', type(exc).__name__)
        cause = exc
        quota_error = False
        while cause:
            if 'RateLimit' in type(cause).__name__ or 'ResourceExhausted' in type(cause).__name__ or getattr(cause, 'code', None) == 429:
                quota_error = True
            cause = cause.__cause__
        yield event('error', text='Gemini quota or rate limit reached. Wait for your quota to reset or increase the Gemini API quota.' if quota_error
                    else 'The AI service could not complete your request. Check API credentials and quota, then try again.')
    yield event('done')


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=6000)
    conversation_id: str | None = None
    language: Language = 'english'


@app.post('/chat')
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(422, 'Please enter a question')
    conv_id, history = await resolve(request.conversation_id, 'text')
    return stream(pipeline(conv_id, request.language, history, question=request.question.strip()))


@app.post('/voice')
async def voice(file: UploadFile = File(...), conversation_id: str | None = Form(None), language: Language = Form('urdu')):
    mime = (file.content_type or '').split(';')[0]
    if mime not in {'audio/webm', 'audio/mp4', 'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/aac', 'audio/ogg', 'audio/flac'}:
        raise HTTPException(415, 'Unsupported recording format. Use WebM, MP4, MP3, WAV, AAC, OGG or FLAC.')
    audio = await file.read(MAX_AUDIO + 1)
    await file.close()
    if not audio or len(audio) > MAX_AUDIO:
        raise HTTPException(413, 'Upload a non-empty recording smaller than 15 MB.')
    conv_id, history = await resolve(conversation_id, 'voice')
    return stream(pipeline(conv_id, language, history, audio=audio, mime=mime))


@app.get('/conversations')
async def conversations(limit: int = Query(50, ge=1, le=100)):
    return {'conversations': await asyncio.to_thread(store.list_conversations, limit)}


@app.get('/conversations/{conversation_id}')
async def conversation(conversation_id: str):
    conv = await asyncio.to_thread(store.get_full_conversation, conversation_id)
    if not conv:
        raise HTTPException(404, 'Conversation not found')
    return conv


@app.delete('/conversations/{conversation_id}')
async def delete(conversation_id: str):
    if not await asyncio.to_thread(store.delete_conversation, conversation_id):
        raise HTTPException(404, 'Conversation not found')
    return {'deleted': True, 'id': conversation_id}


@app.get('/health')
async def health():
    return {'status': 'ok', 'gemini_configured': bool(os.getenv('GEMINI_API_KEY')),
        'uplift_configured': bool(os.getenv('UPLIFTAI_API_KEY')),
        'storage': STORAGE_MODE, 'retrieval': 'qdrant' if QDRANT_URL else 'local'}
