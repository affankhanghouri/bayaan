'use strict';
const $ = selector => document.querySelector(selector);
const messages = $('#messages'), question = $('#question'), status = $('#status');
let conversationId = null, busy = false, recorder = null, microphone = null, timer = null;
const audioUrls = new Set();
function releaseAudio() { for (const url of audioUrls) URL.revokeObjectURL(url); audioUrls.clear(); }
const language = $('#language');
language.value = localStorage.getItem('tafseel-language') || 'english';
language.addEventListener('change', () => localStorage.setItem('tafseel-language', language.value));
function showError(text) { $('#error').textContent = text; $('#error').hidden = !text; }
function setBusy(value) {
  busy = value;
  for (const node of document.querySelectorAll('#send, #record, #new-chat, #language, .suggestions button, .history button')) node.disabled = value;
  question.disabled = value;
  if (!value) status.textContent = '';
}
function resetChat() {
  if (busy || recorder) return;
  conversationId = null;
  messages.replaceChildren(); $('#welcome').hidden = false; showError(''); status.textContent = '';
  releaseAudio();
  refreshHistory(); question.focus();
}
function addMessage(role, text = '') {
  $('#welcome').hidden = true;
  const row = document.createElement('article'); row.className = `message ${role}`;
  const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = role === 'user' ? 'You' : 'ت';
  const content = document.createElement('div'); content.className = 'message-content';
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'You' : 'Tafseel';
  const body = document.createElement('div'); body.className = 'message-text'; body.dir = 'auto'; body.textContent = text;
  content.append(label, body); row.append(avatar, content); messages.append(row); scrollDown();
  return {row, content, body};
}
function scrollDown() { $('#main').scrollTop = $('#main').scrollHeight; }
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail;
    try { const data = await response.json(); detail = typeof data.detail === 'string' ? data.detail : null; } catch {}
    throw new Error(detail || `Request failed (${response.status}). Please try again.`);
  }
  return response;
}
async function refreshHistory() {
  try {
    const data = await (await api('/conversations')).json();
    const history = $('#history'); history.replaceChildren();
    if (!data.conversations.length) {
      const empty = document.createElement('p'); empty.className = 'muted'; empty.textContent = 'Your conversations will appear here.'; history.append(empty);
    }
    for (const conv of data.conversations) {
      const row = document.createElement('div'); row.className = 'history-row' + (conv.id === conversationId ? ' active' : '');
      const open = document.createElement('button'); open.className = 'history-item'; open.textContent = conv.title; open.title = conv.title; open.disabled = busy || !!recorder;
      open.onclick = () => openConversation(conv.id);
      const remove = document.createElement('button'); remove.className = 'delete'; remove.textContent = '×'; remove.setAttribute('aria-label', `Delete ${conv.title}`); remove.disabled = busy || !!recorder;
      remove.onclick = async () => {
        if (busy || recorder) return;
        try { await api(`/conversations/${conv.id}`, {method: 'DELETE'}); if (conv.id === conversationId) resetChat(); else refreshHistory(); } catch (error) { showError(error.message); }
      };
      row.append(open, remove); history.append(row);
    }
  } catch (error) { showError('Could not load conversations. ' + error.message); }
}
async function openConversation(id) {
  if (busy || recorder) return;
  setBusy(true); showError('');
  try {
    const conv = await (await api(`/conversations/${id}`)).json();
    conversationId = id; releaseAudio(); messages.replaceChildren(); $('#welcome').hidden = !!conv.messages.length;
    for (const msg of conv.messages) addMessage(msg.role, msg.content);
    $('#sidebar').classList.remove('open'); $('#menu').setAttribute('aria-expanded', 'false');
  } catch (error) { showError(error.message); }
  finally { setBusy(false); refreshHistory(); }
}
async function consume(response, onEvent) {
  if (!response.body) throw new Error('Streaming is unavailable in this browser.');
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', doneEvent = false;
  try {
    for (;;) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value, {stream: !done});
      buffer = buffer.replace(/\r\n/g, '\n');
      let end;
      while ((end = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
        const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
        if (data) { const event = JSON.parse(data); if (event.type === 'done') doneEvent = true; onEvent(event); }
      }
      if (done) break;
    }
    if (!doneEvent) throw new Error('Connection interrupted before the response finished. Please try again.');
  } finally { reader.releaseLock(); }
}
async function send(text, audio = null) {
  if (busy) return;
  showError(''); setBusy(true);
  let user = addMessage('user', audio ? 'Voice message · transcribing…' : text), assistant = null;
  let sources = [];
  const ensureAssistant = () => assistant || (assistant = addMessage('assistant'));
  try {
    let options;
    if (audio) {
      const form = new FormData();
      const extension = audio.type.includes('mp4') ? 'm4a' : audio.type.includes('ogg') ? 'ogg' : 'webm';
      form.append('file', audio, `recording.${extension}`); form.append('language', language.value);
      if (conversationId) form.append('conversation_id', conversationId);
      options = {method: 'POST', body: form};
    } else {
      options = {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: text, language: language.value, conversation_id: conversationId})};
    }
    const response = await api(audio ? '/voice' : '/chat', options);
    await consume(response, event => {
      switch (event.type) {
        case 'conversation_id': conversationId = event.id; break;
        case 'status': status.textContent = event.text; break;
        case 'transcript': user.body.textContent = event.text; break;
        case 'answer': ensureAssistant().body.textContent = event.text; break;
        case 'token': ensureAssistant().body.textContent += event.text; break;
        case 'sources': sources = event.sources; break;
        case 'audio': {
          const bytes = Uint8Array.from(atob(event.data), c => c.charCodeAt(0));
          const audioUrl = URL.createObjectURL(new Blob([bytes], {type: event.mime_type || 'audio/mpeg'}));
          audioUrls.add(audioUrl);
          const player = document.createElement('audio'); player.controls = true; player.src = audioUrl;
          player.setAttribute('aria-label', 'Listen to Tafseel’s answer'); ensureAssistant().content.append(player);
          player.play().catch(() => { status.textContent = 'Tap play to listen to the answer.'; }); break;
        }
        case 'error': showError(event.text); break;
      }
      scrollDown();
    });
    if (assistant && sources.length) {
      const note = document.createElement('div'); note.className = 'sources'; note.textContent = 'Knowledge base · ' + sources.map(s => s.replace(/\.txt$/, '').replaceAll('_', ' ')).join(' · '); assistant.content.append(note);
    }
  } catch (error) { showError(error.message || 'Connection failed. Please try again.'); }
  finally { setBusy(false); refreshHistory(); question.focus(); }
}
$('#composer').addEventListener('submit', event => {
  event.preventDefault(); const text = question.value.trim();
  if (!text || busy || recorder) return;
  question.value = ''; question.style.height = 'auto'; send(text);
});
question.addEventListener('input', () => { question.style.height = 'auto'; question.style.height = Math.min(question.scrollHeight, 140) + 'px'; });
question.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#composer').requestSubmit(); } });
for (const button of document.querySelectorAll('[data-question]')) button.addEventListener('click', () => { if (!busy && !recorder) send(button.dataset.question); });
$('#new-chat').onclick = resetChat;
$('#menu').onclick = () => { const open = $('#sidebar').classList.toggle('open'); $('#menu').setAttribute('aria-expanded', String(open)); };
$('#main').onclick = () => { $('#sidebar').classList.remove('open'); $('#menu').setAttribute('aria-expanded', 'false'); };
function releaseMic() { clearTimeout(timer); microphone?.getTracks().forEach(track => track.stop()); microphone = null; recorder = null; $('#record').classList.remove('recording'); $('#record').setAttribute('aria-label', 'Start recording'); }
$('#record').onclick = async () => {
  if (busy) return;
  if (recorder) { if (recorder.state === 'recording') recorder.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { showError('Recording needs a supported browser on localhost or HTTPS.'); return; }
  showError(''); $('#record').disabled = true;
  try {
    microphone = await navigator.mediaDevices.getUserMedia({audio: true});
    const mime = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm', 'audio/ogg;codecs=opus'].find(type => MediaRecorder.isTypeSupported(type));
    if (!mime) throw new Error('This browser does not support a compatible recording format.');
    recorder = new MediaRecorder(microphone, {mimeType: mime});
    const chunks = []; let recordingFailed = false;
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onerror = () => { recordingFailed = true; releaseMic(); setBusy(false); showError('Recording failed. Please try again.'); };
    recorder.onstop = () => {
      const audio = new Blob(chunks, {type: mime}); releaseMic(); setBusy(false);
      if (recordingFailed) return;
      if (!audio.size) { showError('No audio recorded. Please try again.'); return; }
      send('', audio);
    };
    recorder.start(); $('#record').classList.add('recording'); $('#record').setAttribute('aria-label', 'Stop recording and send');
    for (const node of document.querySelectorAll('#send, #new-chat, #language, .suggestions button, .history button')) node.disabled = true;
    question.disabled = true;
    status.textContent = 'Recording… Tap the microphone to send. Maximum 60 seconds.';
    timer = setTimeout(() => { if (recorder?.state === 'recording') recorder.stop(); }, 60000);
  } catch (error) { releaseMic(); setBusy(false); showError(error.name === 'NotAllowedError' ? 'Microphone access was denied. Allow it in your browser’s site settings, then try again.' : error.message); }
  finally { $('#record').disabled = false; }
};
window.addEventListener('pagehide', () => { if (recorder?.state === 'recording') { recorder.onstop = null; recorder.stop(); } releaseMic(); releaseAudio(); });
refreshHistory();
