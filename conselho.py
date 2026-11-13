#!/usr/bin/env python3
"""Conselho de Modelos de IA — v0.3"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator
from dotenv import load_dotenv
from anthropic import Anthropic
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.live import Live
from rich.table import Table

load_dotenv(Path(__file__).parent / ".env")
console = Console()

ROLE_DESCRIPTIONS = {
    "propositor": (
        "Você é o especialista técnico do conselho. "
        "Analise o problema com profundidade e proponha uma solução clara e fundamentada."
    ),
    "critico": (
        "Você é o advogado do diabo do conselho. "
        "Questione premissas, identifique riscos e falhas nos argumentos. Seja rigoroso e construtivo."
    ),
    "arbitro": (
        "Você é o pragmático do conselho. "
        "Traga perspectiva prática e independente. Medeie posições e destaque o que realmente importa."
    ),
}

SYNTHESIZER_SYSTEM = """Você é Claude, o árbitro-final e juiz do Conselho de Modelos de IA.
Após o debate entre os membros, produza o seguinte documento estruturado:

## 1. Síntese do Debate

- **Consenso** — pontos em que todos (ou a maioria) concordaram
- **Divergências** — onde as visões diferiram e por quê

## 2. Recomendação Coletiva

Decisão clara e acionável extraída do que o conselho produziu.

## 3. Veredicto de Claude

Aqui você fala por si mesmo — não como síntese, mas como Claude avaliando o debate com sua própria perspectiva:
- O que os membros acertaram
- O que foi ignorado, subestimado ou analisado de forma superficial
- Sua recomendação pessoal final, mesmo que divirja do consenso
- Uma nota de 0 a 10 para a qualidade geral do debate, com justificativa breve

Seja direto e honesto. Este é seu julgamento pessoal, não uma síntese diplomática."""


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Member:
    id: str
    name: str
    role: str
    provider: str
    model: str
    color: str


@dataclass
class Turn:
    member_name: str
    round: int
    content: str


def build_client(provider: str):
    if provider == "anthropic":
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "openrouter":
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://conselho-ia.local"},
        )
    if provider == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError(f"Provider desconhecido: {provider}")


def build_system(member: Member, round_num: int, total_rounds: int) -> str:
    base = f"Você é {member.name}. {ROLE_DESCRIPTIONS.get(member.role, 'Você é um especialista do conselho.')}"
    if round_num == 1:
        return base + "\n\nRodada inicial — responda à tarefa diretamente com sua análise e posição."
    return (
        base
        + f"\n\nRodada {round_num}/{total_rounds} — leia os argumentos anteriores, "
        "critique o que discorda e refine sua posição."
    )


def build_context(task: str, history: list[Turn], round_num: int) -> str:
    parts = [f"**Tarefa:** {task}"]
    if history:
        parts.append("\n---\n**Debate até agora:**\n")
        for t in history:
            parts.append(f"### {t.member_name} (Rodada {t.round})\n{t.content}")
    if round_num > 1:
        parts.append("\n---\nForneça sua resposta para esta rodada.")
    return "\n\n".join(parts)


def token_stream(
    client,
    member: Member,
    system: str,
    user_content: str,
    usage_out: TokenUsage,
) -> Generator[str, None, None]:
    messages = [{"role": "user", "content": user_content}]
    if member.provider == "anthropic":
        with client.messages.stream(
            model=member.model, max_tokens=2048, system=system, messages=messages
        ) as s:
            yield from s.text_stream
            msg = s.get_final_message()
            usage_out.input_tokens += msg.usage.input_tokens
            usage_out.output_tokens += msg.usage.output_tokens
    else:
        for chunk in client.chat.completions.create(
            model=member.model,
            max_tokens=2048,
            stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "system", "content": system}] + messages,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if getattr(chunk, "usage", None):
                usage_out.input_tokens += chunk.usage.prompt_tokens or 0
                usage_out.output_tokens += chunk.usage.completion_tokens or 0


def show_stream(gen: Generator[str, None, None], title: str, color: str) -> str:
    full = ""
    with Live(
        Panel("", title=title, border_style=color, padding=(1, 2)),
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
    ) as live:
        for token in gen:
            full += token
            live.update(Panel(full, title=title, border_style=color, padding=(1, 2)))
    return full


def call_member(
    client,
    member: Member,
    system: str,
    user_content: str,
    usage_out: TokenUsage,
    retries: int = 3,
) -> str:
    title = f"[{member.color}][bold]{member.name}[/bold][/{member.color}]"
    for attempt in range(retries):
        try:
            return show_stream(
                token_stream(client, member, system, user_content, usage_out),
                title,
                member.color,
            )
        except Exception as exc:
            if "429" in str(exc) and attempt < retries - 1:
                wait = 30 * (attempt + 1)
                console.print(f"  [dim]{member.name} rate-limited — aguardando {wait}s...[/dim]")
                time.sleep(wait)
            else:
                console.print(f"  [red]Erro {member.name}: {exc}[/red]")
                return ""
    return ""


def synthesize(
    task: str, history: list[Turn], model: str, provider: str
) -> tuple[str, TokenUsage]:
    debate = "\n\n".join(
        f"### {t.member_name} (Rodada {t.round})\n{t.content}" for t in history
    )
    user_content = (
        f"**Tarefa:** {task}\n\n---\n**Debate:**\n\n{debate}"
        "\n\n---\nProduz a síntese e o veredicto final."
    )
    title = "[bold yellow]Veredicto de Claude[/bold yellow]"
    full = ""
    usage = TokenUsage()

    with Live(
        Panel("", title=title, border_style="yellow", padding=(1, 2)),
        console=console,
        refresh_per_second=15,
        vertical_overflow="visible",
    ) as live:
        if provider == "anthropic":
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            with client.messages.stream(
                model=model,
                max_tokens=3000,
                system=SYNTHESIZER_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            ) as s:
                for text in s.text_stream:
                    full += text
                    live.update(Panel(full, title=title, border_style="yellow", padding=(1, 2)))
                msg = s.get_final_message()
                usage.input_tokens = msg.usage.input_tokens
                usage.output_tokens = msg.usage.output_tokens
        else:
            client = build_client(provider)
            for chunk in client.chat.completions.create(
                model=model,
                max_tokens=3000,
                stream=True,
                stream_options={"include_usage": True},
                messages=[
                    {"role": "system", "content": SYNTHESIZER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full += delta
                    live.update(Panel(full, title=title, border_style="yellow", padding=(1, 2)))
                if getattr(chunk, "usage", None):
                    usage.input_tokens = chunk.usage.prompt_tokens or 0
                    usage.output_tokens = chunk.usage.completion_tokens or 0

    return full, usage


def show_token_table(
    members: list[Member], member_usages: dict[str, TokenUsage], synth_usage: TokenUsage
) -> None:
    table = Table(title="Resumo de Tokens", border_style="dim", show_lines=True)
    table.add_column("Membro", style="bold")
    table.add_column("Papel")
    table.add_column("Modelo", style="dim")
    table.add_column("Entrada", justify="right")
    table.add_column("Saída", justify="right")
    table.add_column("Total", justify="right", style="bold")

    total_in = total_out = 0
    for m in members:
        u = member_usages.get(m.id, TokenUsage())
        table.add_row(m.name, m.role, m.model, str(u.input_tokens), str(u.output_tokens), str(u.total))
        total_in += u.input_tokens
        total_out += u.output_tokens

    table.add_row(
        "Claude", "síntese/juiz", "claude-sonnet-4-6",
        str(synth_usage.input_tokens), str(synth_usage.output_tokens), str(synth_usage.total),
    )
    total_in += synth_usage.input_tokens
    total_out += synth_usage.output_tokens

    table.add_section()
    table.add_row(
        "[bold]Total[/bold]", "", "",
        f"[bold]{total_in}[/bold]",
        f"[bold]{total_out}[/bold]",
        f"[bold]{total_in + total_out}[/bold]",
    )
    console.print()
    console.print(table)


def save_transcript(
    task: str,
    history: list[Turn],
    synthesis: str,
    members: list[Member],
    rounds: int,
    member_usages: dict[str, TokenUsage],
    synth_usage: TokenUsage,
    output_dir: Path,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"debate_{ts}.md"

    member_list = " · ".join(f"{m.name} ({m.role})" for m in members)

    lines = [
        "# Conselho de Modelos de IA\n",
        f"**Tarefa:** {task}\n",
        f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        f"**Membros:** {member_list}",
        f"**Rodadas:** {rounds}\n",
    ]

    cur_round = 0
    for t in history:
        if t.round != cur_round:
            cur_round = t.round
            lines.append(f"\n---\n\n## Rodada {cur_round}\n")
        lines.append(f"### {t.member_name}\n\n{t.content}\n")

    lines.append(f"\n---\n\n## Síntese e Veredicto Final\n\n{synthesis}\n")

    lines.append("\n---\n\n## Resumo de Tokens\n")
    lines.append("| Membro | Papel | Modelo | Entrada | Saída | Total |")
    lines.append("|--------|-------|--------|---------|-------|-------|")
    total_in = total_out = 0
    for m in members:
        u = member_usages.get(m.id, TokenUsage())
        lines.append(f"| {m.name} | {m.role} | `{m.model}` | {u.input_tokens} | {u.output_tokens} | {u.total} |")
        total_in += u.input_tokens
        total_out += u.output_tokens
    lines.append(
        f"| Claude | síntese/juiz | `claude-sonnet-4-6` | "
        f"{synth_usage.input_tokens} | {synth_usage.output_tokens} | {synth_usage.total} |"
    )
    total_in += synth_usage.input_tokens
    total_out += synth_usage.output_tokens
    lines.append(f"| **Total** | | | **{total_in}** | **{total_out}** | **{total_in + total_out}** |")

    path.write_text("\n".join(lines))
    return path


def run_council(task: str, config: dict, save: bool = False) -> str:
    members = [Member(**m) for m in config["members"]]
    rounds = config.get("rounds", 2)
    member_delay = config.get("member_delay", 0)
    synth_model = config.get("synthesizer_model", "claude-sonnet-4-6")
    synth_provider = config.get("synthesizer_provider", "anthropic")
    clients = {m.id: build_client(m.provider) for m in members}
    history: list[Turn] = []
    member_usages: dict[str, TokenUsage] = {m.id: TokenUsage() for m in members}

    member_list = " · ".join(
        f"[{m.color}]{m.name}[/{m.color}] [dim]({m.role})[/dim]" for m in members
    )
    console.print()
    console.print(Rule("[bold]⚖ Conselho de Modelos de IA[/bold]"))
    console.print(f"\n  [bold]Tarefa:[/bold] {task[:120]}{'...' if len(task) > 120 else ''}")
    delay_info = f" · delay {member_delay}s entre membros" if member_delay else ""
    console.print(f"  [dim]{member_list} · {rounds} rodada{'s' if rounds > 1 else ''}{delay_info}[/dim]\n")

    for round_num in range(1, rounds + 1):
        if rounds > 1:
            console.print(Rule(f"[dim]Rodada {round_num} de {rounds}[/dim]"))
        for i, member in enumerate(members):
            if member_delay and i > 0:
                console.print(f"  [dim]aguardando {member_delay}s antes de chamar {member.name}...[/dim]")
                time.sleep(member_delay)
            system = build_system(member, round_num, rounds)
            user_content = build_context(task, history, round_num)
            response = call_member(
                clients[member.id], member, system, user_content, member_usages[member.id]
            )
            if response:
                history.append(Turn(member_name=member.name, round=round_num, content=response))

    if not history:
        console.print("\n  [red]Nenhum membro respondeu — verifique as chaves de API.[/red]\n")
        return ""

    console.print()
    console.print(Rule("[bold yellow]⚖ Veredicto Final de Claude[/bold yellow]"))
    synthesis, synth_usage = synthesize(task, history, synth_model, synth_provider)

    show_token_table(members, member_usages, synth_usage)

    if save:
        output_dir = Path(os.environ.get("CONSELHO_DEBATES_DIR", Path(__file__).parent / "debates"))
        output_dir.mkdir(parents=True, exist_ok=True)
        path = save_transcript(
            task, history, synthesis, members, rounds,
            member_usages, synth_usage, output_dir,
        )
        console.print(f"\n  [dim]Transcript salvo em: {path}[/dim]")

    console.print()
    return synthesis


def main():
    parser = argparse.ArgumentParser(
        description="Conselho de Modelos de IA — debate autônomo multi-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            '  python conselho.py "melhor estratégia de backup?"\n'
            '  python conselho.py --rounds 1 "explique JWT em 3 linhas"\n'
            '  cat script.py | python conselho.py "revise este código"\n'
            '  python conselho.py --file docker-compose.yml "este arquivo tem problemas?"\n'
            "  python conselho.py -i  # modo interativo"
        ),
    )
    parser.add_argument("tarefa", nargs="?", help="Tarefa (use - para ler do stdin)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--rounds", "-r", type=int, help="Número de rodadas")
    parser.add_argument("--file", "-f", metavar="PATH", help="Incluir arquivo como contexto")
    parser.add_argument("--save", "-s", action="store_true", help="Salvar transcript em /debates/")
    parser.add_argument("--interactive", "-i", action="store_true", help="Modo interativo (loop)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.rounds:
        config["rounds"] = args.rounds

    piped = ""
    if not sys.stdin.isatty() and not args.interactive:
        piped = sys.stdin.read().strip()

    def with_context(task: str) -> str:
        if args.file:
            ctx = Path(args.file).read_text()
            return f"{task}\n\n```\n{ctx}\n```"
        if piped and args.tarefa != "-":
            return f"{task}\n\n```\n{piped}\n```"
        return task

    if args.interactive:
        console.print(
            "\n[bold]⚖ Conselho de Modelos de IA[/bold]  "
            "[dim]— modo interativo (Ctrl+C para sair)[/dim]\n"
        )
        while True:
            try:
                task = console.input("[bold cyan]Tarefa:[/bold cyan] ").strip()
                if not task or task.lower() in ("sair", "exit", "quit"):
                    console.print("[dim]Até logo.[/dim]\n")
                    break
                run_council(with_context(task), config, save=args.save)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Até logo.[/dim]\n")
                break
        return

    if args.tarefa == "-":
        task = piped
    elif args.tarefa:
        task = with_context(args.tarefa)
    elif piped:
        task = piped
    else:
        console.print("[bold]⚖ Conselho de Modelos de IA[/bold]  [dim](Ctrl+C para sair)[/dim]")
        try:
            task = console.input("\n  [bold]Tarefa:[/bold] ").strip()
            task = with_context(task)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Abortado.[/dim]")
            sys.exit(0)

    if not task:
        console.print("[red]Tarefa vazia.[/red]")
        sys.exit(1)

    run_council(task, config, save=args.save)


if __name__ == "__main__":
    main()
