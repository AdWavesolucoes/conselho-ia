#!/usr/bin/env python3
"""Conselho de IA — Web UI"""

import os
import json
import queue
import re
import threading
from pathlib import Path
from typing import AsyncGenerator
import asyncio

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

load_dotenv(Path(__file__).parent / ".env")

from conselho import (
    Member, Turn, TokenUsage,
    build_client, build_system, build_context,
    token_stream, save_transcript, SYNTHESIZER_SYSTEM,
)
from anthropic import Anthropic

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEBATES_DIR = Path(os.environ.get("CONSELHO_DEBATES_DIR", Path(__file__).parent / "debates"))

app = FastAPI(title="Conselho de IA")


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def run_council_streamed(task: str, config: dict, save: bool, q: queue.Queue):
    def put(event):
        q.put(event)

    try:
        members = [Member(**m) for m in config["members"]]
        rounds = config.get("rounds", 2)
        synth_model = config.get("synthesizer_model", "claude-sonnet-4-6")
        synth_provider = config.get("synthesizer_provider", "anthropic")
        clients = {m.id: build_client(m.provider) for m in members}
        history: list[Turn] = []
        member_usages: dict[str, TokenUsage] = {m.id: TokenUsage() for m in members}

        put({"type": "council_start", "rounds": rounds, "members": [
            {"name": m.name, "role": m.role, "color": m.color} for m in members
        ]})

        for round_num in range(1, rounds + 1):
            put({"type": "round_start", "round": round_num, "total": rounds})

            for member in members:
                system = build_system(member, round_num, rounds)
                user_content = build_context(task, history, round_num)
                usage = member_usages[member.id]

                put({"type": "member_start", "name": member.name, "role": member.role, "color": member.color})

                full = ""
                try:
                    for token in token_stream(clients[member.id], member, system, user_content, usage):
                        full += token
                        put({"type": "token", "text": token})
                except Exception as exc:
                    put({"type": "member_error", "name": member.name, "error": str(exc)})

                put({"type": "member_done", "name": member.name,
                     "usage": {"input": usage.input_tokens, "output": usage.output_tokens}})

                if full:
                    history.append(Turn(member_name=member.name, round=round_num, content=full))

        if not history:
            put({"type": "error", "message": "Nenhum membro respondeu — verifique as chaves de API."})
            put({"type": "done"})
            return

        # Synthesis
        put({"type": "synthesis_start"})

        debate_text = "\n\n".join(
            f"### {t.member_name} (Rodada {t.round})\n{t.content}" for t in history
        )
        user_content = (
            f"**Tarefa:** {task}\n\n---\n**Debate:**\n\n{debate_text}"
            "\n\n---\nProduz a síntese e o veredicto final."
        )

        synth_usage = TokenUsage()
        synthesis_full = ""

        try:
            if synth_provider == "anthropic":
                client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                with client.messages.stream(
                    model=synth_model, max_tokens=3000, system=SYNTHESIZER_SYSTEM,
                    messages=[{"role": "user", "content": user_content}],
                ) as s:
                    for text in s.text_stream:
                        synthesis_full += text
                        put({"type": "synthesis_token", "text": text})
                    msg = s.get_final_message()
                    synth_usage.input_tokens = msg.usage.input_tokens
                    synth_usage.output_tokens = msg.usage.output_tokens
            else:
                client = build_client(synth_provider)
                for chunk in client.chat.completions.create(
                    model=synth_model, max_tokens=3000, stream=True,
                    stream_options={"include_usage": True},
                    messages=[
                        {"role": "system", "content": SYNTHESIZER_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        synthesis_full += delta
                        put({"type": "synthesis_token", "text": delta})
                    if getattr(chunk, "usage", None):
                        synth_usage.input_tokens = chunk.usage.prompt_tokens or 0
                        synth_usage.output_tokens = chunk.usage.completion_tokens or 0
        except Exception as exc:
            put({"type": "error", "message": f"Síntese falhou: {exc}"})

        put({"type": "synthesis_done",
             "usage": {"input": synth_usage.input_tokens, "output": synth_usage.output_tokens}})

        # Token summary
        rows = []
        total_in = total_out = 0
        for m in members:
            u = member_usages.get(m.id, TokenUsage())
            rows.append({"name": m.name, "role": m.role,
                         "input": u.input_tokens, "output": u.output_tokens})
            total_in += u.input_tokens
            total_out += u.output_tokens
        rows.append({"name": "Claude", "role": "síntese/juiz",
                     "input": synth_usage.input_tokens, "output": synth_usage.output_tokens})
        total_in += synth_usage.input_tokens
        total_out += synth_usage.output_tokens

        put({"type": "tokens_summary", "rows": rows, "total_in": total_in, "total_out": total_out})

        if save and synthesis_full:
            DEBATES_DIR.mkdir(parents=True, exist_ok=True)
            path = save_transcript(
                task, history, synthesis_full, members, rounds,
                member_usages, synth_usage, DEBATES_DIR,
            )
            put({"type": "saved", "path": str(path)})

    except Exception as exc:
        put({"type": "error", "message": str(exc)})

    put({"type": "done"})


async def event_stream(task: str, config: dict, save: bool) -> AsyncGenerator[str, None]:
    q: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    thread = threading.Thread(
        target=run_council_streamed, args=(task, config, save, q), daemon=True
    )
    thread.start()

    while True:
        try:
            event = await loop.run_in_executor(None, lambda: q.get(timeout=0.3))
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") == "done":
                break
        except queue.Empty:
            yield "data: {\"type\":\"ping\"}\n\n"

    thread.join(timeout=10)


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


@app.post("/debate")
async def debate(request: Request):
    body = await request.json()
    task = body.get("task", "").strip()
    if not task:
        return JSONResponse({"error": "Tarefa vazia"}, status_code=400)

    config = load_config()
    if "rounds" in body:
        config["rounds"] = int(body["rounds"])
    save = body.get("save", True)

    return StreamingResponse(
        event_stream(task, config, save),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/debates")
async def list_debates():
    if not DEBATES_DIR.exists():
        return JSONResponse([])
    files = sorted(DEBATES_DIR.glob("debate_*.md"), reverse=True)
    return JSONResponse([f.name for f in files])


@app.get("/debates/{filename}")
async def get_debate(filename: str):
    if not re.fullmatch(r"debate_[\w.-]+\.md", filename):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = DEBATES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"content": path.read_text()})


HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>⚖ Conselho de IA</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0c0c0c;color:#d4d4d4;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}

header{padding:20px 32px;border-bottom:1px solid #1e1e1e;display:flex;align-items:center;gap:12px}
header h1{font-size:1.25rem;color:#fff;font-weight:600}
header p{color:#555;font-size:0.8rem;margin-top:2px}
.header-info{flex:1}

.container{max-width:1280px;margin:0 auto;padding:28px 32px}

/* Form */
.form-card{background:#141414;border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:28px}
textarea{width:100%;background:#0c0c0c;border:1px solid #2a2a2a;border-radius:8px;padding:12px 14px;color:#d4d4d4;font-size:0.95rem;resize:vertical;min-height:90px;font-family:inherit;line-height:1.5;transition:border-color .15s}
textarea:focus{outline:none;border-color:#404040}
.form-row{display:flex;gap:12px;align-items:center;margin-top:14px;flex-wrap:wrap}
.form-label{color:#666;font-size:0.82rem;display:flex;align-items:center;gap:6px}
input[type=number]{background:#0c0c0c;border:1px solid #2a2a2a;border-radius:6px;padding:5px 8px;color:#d4d4d4;font-size:0.85rem;width:56px;text-align:center}
input[type=checkbox]{accent-color:#3b82f6;width:14px;height:14px;cursor:pointer}
.btn{background:#2563eb;color:#fff;border:none;border-radius:8px;padding:9px 22px;font-size:0.9rem;cursor:pointer;transition:background .15s;display:flex;align-items:center;gap:8px;margin-left:auto}
.btn:hover{background:#1d4ed8}
.btn:disabled{background:#1e1e1e;color:#444;cursor:not-allowed}
.hint{color:#444;font-size:0.75rem;margin-left:auto}

/* Round */
.round-label{text-align:center;color:#3a3a3a;font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin:24px 0 14px;display:flex;align-items:center;gap:12px}
.round-label::before,.round-label::after{content:'';flex:1;height:1px;background:#1e1e1e}

/* Panels grid */
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}

/* Member panel */
.panel{background:#141414;border-radius:10px;border:1px solid #1e1e1e;overflow:hidden;display:flex;flex-direction:column}
.panel-header{padding:10px 14px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #1a1a1a}
.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.panel-name{font-weight:600;font-size:0.85rem;color:#e0e0e0}
.panel-role{font-size:0.75rem;color:#555;margin-top:1px}
.panel-status{margin-left:auto;font-size:0.72rem;color:#444;font-variant-numeric:tabular-nums}
.panel-body{padding:14px;font-size:0.84rem;line-height:1.7;flex:1;overflow-y:auto;max-height:480px}
.panel-body.active{border-left:2px solid var(--c)}
.empty-body{color:#333;font-style:italic;font-size:0.82rem}

/* Color accents */
.c-cyan{--c:#06b6d4}.c-cyan .dot{background:#06b6d4}.c-cyan .panel{border-top:2px solid #06b6d4}
.c-green{--c:#22c55e}.c-green .dot{background:#22c55e}.c-green .panel{border-top:2px solid #22c55e}
.c-yellow{--c:#eab308}.c-yellow .dot{background:#eab308}.c-yellow .panel{border-top:2px solid #eab308}
.c-blue{--c:#3b82f6}.c-blue .dot{background:#3b82f6}.c-blue .panel{border-top:2px solid #3b82f6}
.c-magenta{--c:#a855f7}.c-magenta .dot{background:#a855f7}.c-magenta .panel{border-top:2px solid #a855f7}
.c-red{--c:#ef4444}.c-red .dot{background:#ef4444}.c-red .panel{border-top:2px solid #ef4444}

/* Synthesis */
.synthesis-wrap{margin-top:20px}
.synthesis-card{background:#111008;border:1px solid #2a2608;border-top:2px solid #ca8a04;border-radius:10px;overflow:hidden}
.synthesis-head{padding:10px 14px;border-bottom:1px solid #1e1a04;display:flex;align-items:center;gap:8px}
.synthesis-head span{font-weight:600;color:#ca8a04;font-size:0.85rem}
.synthesis-body{padding:16px;font-size:0.84rem;line-height:1.7;max-height:700px;overflow-y:auto}
.synthesis-body.active{border-left:2px solid #ca8a04}

/* Token table */
.token-wrap{margin-top:16px;background:#141414;border:1px solid #1e1e1e;border-radius:10px;overflow:hidden}
.token-wrap h3{padding:10px 16px;font-size:0.75rem;color:#444;text-transform:uppercase;letter-spacing:2px;border-bottom:1px solid #1a1a1a}
.token-table{width:100%;border-collapse:collapse;font-size:0.82rem}
.token-table th{padding:8px 16px;text-align:left;color:#444;font-weight:500;border-bottom:1px solid #1a1a1a}
.token-table td{padding:8px 16px;border-bottom:1px solid #161616;color:#888}
.token-table td:first-child{color:#bbb}
.token-table .num{text-align:right;font-variant-numeric:tabular-nums}
.token-table tr.total td{border-bottom:none;color:#d4d4d4;font-weight:600}
.saved-note{margin-top:10px;display:inline-flex;align-items:center;gap:7px;background:#0a1a0a;border:1px solid #1a2e1a;border-radius:7px;padding:7px 12px;font-size:0.78rem;color:#4ade80}

/* Past debates */
.past{margin-top:48px}
.past h2{font-size:0.75rem;color:#3a3a3a;text-transform:uppercase;letter-spacing:3px;margin-bottom:14px}
.debate-list{display:flex;flex-direction:column;gap:6px}
.debate-item{background:#141414;border:1px solid #1e1e1e;border-radius:8px;padding:10px 14px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;transition:border-color .15s}
.debate-item:hover{border-color:#333}
.debate-item .di-date{font-size:0.82rem;color:#666}
.debate-item .di-arrow{color:#333;font-size:0.9rem}

/* Viewer modal */
.viewer{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:200;overflow-y:auto;padding:28px 20px}
.viewer-inner{max-width:860px;margin:0 auto;background:#141414;border-radius:12px;border:1px solid #222;padding:28px}
.viewer-close{float:right;background:#1e1e1e;border:1px solid #2a2a2a;color:#888;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:0.8rem;margin-bottom:16px}
.viewer-close:hover{color:#d4d4d4}
.viewer-body{font-size:0.85rem;line-height:1.75;clear:both}

/* Markdown styles (shared) */
.md h1,.md h2,.md h3{color:#e0e0e0;margin:14px 0 6px}
.md h1{font-size:1.05rem}.md h2{font-size:.97rem}.md h3{font-size:.9rem}
.md p{margin:5px 0}
.md ul,.md ol{padding-left:18px;margin:5px 0}
.md li{margin:2px 0}
.md code{background:#1a1a1a;padding:1px 5px;border-radius:3px;font-size:.82em;color:#7dd3fc;font-family:monospace}
.md pre{background:#0f0f0f;border:1px solid #222;border-radius:6px;padding:12px;overflow-x:auto;margin:8px 0}
.md pre code{background:none;padding:0;color:#d4d4d4}
.md table{border-collapse:collapse;margin:8px 0;font-size:.82rem;width:100%}
.md td,.md th{border:1px solid #222;padding:5px 10px}
.md th{background:#1a1a1a;color:#888}
.md strong{color:#e0e0e0}
.md em{color:#888}
.md blockquote{border-left:2px solid #333;padding-left:12px;color:#666;margin:6px 0}
.md hr{border:none;border-top:1px solid #1e1e1e;margin:14px 0}
.md a{color:#3b82f6;text-decoration:none}

/* synthesis md overrides */
.synthesis-body.md strong{color:#fef08a}
.synthesis-body.md h1,.synthesis-body.md h2,.synthesis-body.md h3{color:#fef08a}
.synthesis-body.md blockquote{border-left-color:#ca8a04}

/* spinner */
.spin{display:inline-block;width:12px;height:12px;border:1.5px solid #333;border-top-color:#666;border-radius:50%;animation:s .7s linear infinite;vertical-align:middle}
@keyframes s{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<header>
  <div class="header-info">
    <h1>⚖ Conselho de IA</h1>
    <p>Nemotron Ultra · Nemotron Super · Nemotron Nano · Claude (juiz)</p>
  </div>
</header>

<div class="container">

  <div class="form-card">
    <textarea id="task" placeholder="Digite a tarefa ou pergunta para o conselho debater... (Ctrl+Enter para enviar)"></textarea>
    <div class="form-row">
      <label class="form-label">Rodadas <input type="number" id="rounds" value="2" min="1" max="5"></label>
      <label class="form-label"><input type="checkbox" id="save" checked> Salvar no Obsidian</label>
      <span class="hint">Ctrl+Enter</span>
      <button class="btn" id="btn" onclick="startDebate()">⚖ Debater</button>
    </div>
  </div>

  <div id="debateArea"></div>

  <div class="past">
    <h2>Debates anteriores</h2>
    <div class="debate-list" id="debateList"><p style="color:#333;font-size:.82rem">carregando...</p></div>
  </div>

</div>

<div class="viewer" id="viewer">
  <div class="viewer-inner">
    <button class="viewer-close" onclick="closeViewer()">✕ fechar</button>
    <div class="viewer-body md" id="viewerBody"></div>
  </div>
</div>

<script>
marked.setOptions({breaks:true});
function md(t){return marked.parse(t||'')}

let streamBodyId = null;
let streamRaw = '';
let synthRaw = '';
let currentRound = 0;

function startDebate(){
  const task = document.getElementById('task').value.trim();
  if(!task) return;
  const rounds = parseInt(document.getElementById('rounds').value)||2;
  const save = document.getElementById('save').checked;
  const btn = document.getElementById('btn');

  // reset
  streamBodyId = null; streamRaw = ''; synthRaw = ''; currentRound = 0;
  document.getElementById('debateArea').innerHTML = '';

  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> Debatendo...';

  fetch('/debate',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task, rounds, save})
  }).then(r=>{
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    function pump(){
      reader.read().then(({done,value})=>{
        if(done){finish();return;}
        buf += dec.decode(value,{stream:true});
        const lines = buf.split('\n');
        buf = lines.pop();
        for(const line of lines){
          if(line.startsWith('data: ')){
            try{ handle(JSON.parse(line.slice(6))); }catch(e){}
          }
        }
        pump();
      }).catch(()=>finish());
    }
    pump();
    function finish(){btn.disabled=false;btn.innerHTML='⚖ Debater';}
  });
}

function handle(ev){
  const area = document.getElementById('debateArea');
  switch(ev.type){
    case 'ping': break;

    case 'round_start':{
      currentRound = ev.round;
      const div = document.createElement('div');
      div.innerHTML = `
        <div class="round-label">Rodada ${ev.round} de ${ev.total}</div>
        <div class="panels" id="r${ev.round}"></div>`;
      area.appendChild(div);
      break;
    }

    case 'member_start':{
      streamRaw = '';
      const key = `${currentRound}_${slug(ev.name)}`;
      streamBodyId = 'b_'+key;
      const grid = document.getElementById('r'+currentRound);
      const wrap = document.createElement('div');
      wrap.className = `c-${ev.color}`;
      wrap.innerHTML = `
        <div class="panel" id="p_${key}">
          <div class="panel-header">
            <div class="dot"></div>
            <div><div class="panel-name">${ev.name}</div><div class="panel-role">${ev.role}</div></div>
            <div class="panel-status" id="s_${key}"><span class="spin"></span></div>
          </div>
          <div class="panel-body md active" id="${streamBodyId}" style="--c:var(--c)"></div>
        </div>`;
      grid.appendChild(wrap);
      break;
    }

    case 'token':{
      streamRaw += ev.text;
      const el = document.getElementById(streamBodyId);
      if(el){el.innerHTML=md(streamRaw);el.scrollTop=el.scrollHeight;}
      break;
    }

    case 'member_done':{
      const key = `${currentRound}_${slug(ev.name)}`;
      const s = document.getElementById('s_'+key);
      const b = document.getElementById('b_'+key);
      if(s) s.innerHTML = ev.usage.input||ev.usage.output
        ? `${fmt(ev.usage.input+ev.usage.output)} tok` : '✓';
      if(b) b.classList.remove('active');
      break;
    }

    case 'member_error':{
      const key = `${currentRound}_${slug(ev.name)}`;
      const s = document.getElementById('s_'+key);
      const b = document.getElementById('b_'+key);
      if(s) s.innerHTML='<span style="color:#ef4444">✗</span>';
      if(b){b.classList.remove('active');b.innerHTML='<span class="empty-body">sem resposta</span>';}
      break;
    }

    case 'synthesis_start':{
      synthRaw = '';
      const el = document.createElement('div');
      el.className = 'synthesis-wrap';
      el.innerHTML = `
        <div class="synthesis-card">
          <div class="synthesis-head">
            <span>⚖ Veredicto de Claude</span>
            <span id="synthStatus" style="margin-left:auto;font-size:.72rem;color:#555"><span class="spin"></span></span>
          </div>
          <div class="synthesis-body md active" id="synthBody"></div>
        </div>
        <div id="tokenWrap"></div>
        <div id="savedNote"></div>`;
      area.appendChild(el);
      el.scrollIntoView({behavior:'smooth'});
      break;
    }

    case 'synthesis_token':{
      synthRaw += ev.text;
      const el = document.getElementById('synthBody');
      if(el){el.innerHTML=md(synthRaw);el.scrollTop=el.scrollHeight;}
      break;
    }

    case 'synthesis_done':{
      const s = document.getElementById('synthStatus');
      const b = document.getElementById('synthBody');
      if(s) s.textContent = `${fmt(ev.usage.input+ev.usage.output)} tok`;
      if(b) b.classList.remove('active');
      break;
    }

    case 'tokens_summary':{
      const rows = ev.rows.map(r=>`
        <tr>
          <td>${r.name}</td>
          <td style="color:#555">${r.role}</td>
          <td class="num">${r.input.toLocaleString('pt-BR')}</td>
          <td class="num">${r.output.toLocaleString('pt-BR')}</td>
          <td class="num">${(r.input+r.output).toLocaleString('pt-BR')}</td>
        </tr>`).join('');
      const total = `
        <tr class="total">
          <td>Total</td><td></td>
          <td class="num">${ev.total_in.toLocaleString('pt-BR')}</td>
          <td class="num">${ev.total_out.toLocaleString('pt-BR')}</td>
          <td class="num">${(ev.total_in+ev.total_out).toLocaleString('pt-BR')}</td>
        </tr>`;
      document.getElementById('tokenWrap').innerHTML = `
        <div class="token-wrap">
          <h3>Tokens</h3>
          <table class="token-table">
            <thead><tr><th>Membro</th><th>Papel</th><th class="num">Entrada</th><th class="num">Saída</th><th class="num">Total</th></tr></thead>
            <tbody>${rows}${total}</tbody>
          </table>
        </div>`;
      break;
    }

    case 'saved':{
      const name = ev.path.split('/').pop();
      document.getElementById('savedNote').innerHTML =
        `<div class="saved-note">✓ Salvo no Obsidian — ${name}</div>`;
      loadDebates();
      break;
    }

    case 'error':{
      area.insertAdjacentHTML('beforeend',
        `<p style="color:#ef4444;font-size:.85rem;margin-top:12px">✗ ${ev.message}</p>`);
      break;
    }

    case 'done':{
      document.getElementById('btn').disabled = false;
      document.getElementById('btn').innerHTML = '⚖ Debater';
      break;
    }
  }
}

function slug(s){return s.replace(/\s+/g,'-').replace(/[^a-zA-Z0-9-]/g,'')}
function fmt(n){return n>=1000?(n/1000).toFixed(1)+'k':String(n)}

function loadDebates(){
  fetch('/debates').then(r=>r.json()).then(files=>{
    const list = document.getElementById('debateList');
    if(!files.length){list.innerHTML='<p style="color:#333;font-size:.82rem">nenhum debate salvo ainda</p>';return;}
    list.innerHTML = files.map(f=>{
      const m = f.match(/debate_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
      const label = m ? `${m[3]}/${m[2]}/${m[1]} ${m[4]}:${m[5]}` : f;
      return `<div class="debate-item" onclick="openDebate('${f}')">
        <span class="di-date">${label}</span>
        <span class="di-arrow">→</span>
      </div>`;
    }).join('');
  });
}

function openDebate(f){
  fetch('/debates/'+f).then(r=>r.json()).then(data=>{
    document.getElementById('viewerBody').innerHTML = md(data.content);
    document.getElementById('viewer').style.display = 'block';
    document.body.style.overflow = 'hidden';
  });
}
function closeViewer(){
  document.getElementById('viewer').style.display='none';
  document.body.style.overflow='';
}

document.addEventListener('keydown',e=>{
  if(e.key==='Escape') closeViewer();
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter') startDebate();
});

loadDebates();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("CONSELHO_HOST", "127.0.0.1"),
                port=int(os.environ.get("CONSELHO_PORT", "7860")), log_level="warning")
