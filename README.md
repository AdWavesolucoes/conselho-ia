# Conselho de IA

Debate estruturado entre vários LLMs, com papéis fixos e um juiz final.

Uma pergunta difícil vai para três modelos ao mesmo tempo. Cada um tem um papel
fixo: **propositor** (propõe a solução), **crítico** (ataca as premissas) e
**árbitro** (puxa para o lado prático). Eles debatem por N rodadas, e cada
rodada enxerga as falas da anterior. No fim, um modelo mais forte lê o debate
inteiro e escreve a síntese: consenso, divergências, recomendação e um veredito
próprio com nota.

```
            ┌──────────────┐
 tarefa ──▶ │  rodada 1..N │   propositor ─┐
            │              │   crítico    ─┼─▶ histórico compartilhado
            │              │   árbitro    ─┘
            └──────┬───────┘
                   ▼
            sintetizador ──▶ consenso · divergências · recomendação · veredito
```

## Por que existe

Uma resposta isolada de um LLM vem com confiança uniforme, seja ela certa ou
errada. Colocar um modelo no papel de crítico força as premissas a aparecerem,
e o sintetizador separa o que foi consenso do que foi só afirmação. Usei o
conselho em decisões de infraestrutura do meu homelab, como estratégia de
volumes Docker e escolha de stack.

## Detalhes

- **Vários provedores no mesmo debate:** Anthropic, OpenAI, OpenRouter e Ollama
  local, todos pela mesma interface (`build_client`). Os membros e os papéis
  ficam em `config.yaml`. A configuração padrão usa modelos gratuitos do
  OpenRouter nos membros e paga só o sintetizador.
- **Streaming:** no terminal, com `rich`, e no navegador, por SSE (`web.py`,
  FastAPI).
- **Custo visível:** ao final, uma tabela mostra os tokens de entrada e saída
  por membro e do sintetizador.
- **Transcript:** com `--save`, o debate é salvo em Markdown para consulta
  depois. A interface web lista os debates salvos.

## Uso

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha só as chaves dos provedores do config.yaml

# terminal
python conselho.py "Vale migrar de SQLite para Postgres num app com 3 usuários?" -r 2 --save
python conselho.py -f docker-compose.yml "O que está frágil nesta stack?"
python conselho.py -i                      # modo interativo

# navegador (http://127.0.0.1:7860)
python web.py
```

| Variável | Padrão | Uso |
|---|---|---|
| `CONSELHO_DEBATES_DIR` | `./debates` | onde os transcripts são salvos |
| `CONSELHO_HOST` / `CONSELHO_PORT` | `127.0.0.1` / `7860` | bind da interface web |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | membros locais |

> A interface web não tem autenticação. Deixe em `127.0.0.1` ou coloque atrás
> de um proxy com login.

## Licença

MIT
