"""Unified Click & Rich CLI entry point for MicroGen serving, interactive terminal chat, and benchmarking."""

import sys
import time
import json
import threading
from typing import Optional
import click
import uvicorn

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import torch
from transformers import AutoTokenizer

from microgen.sdk.engine import LLMEngine
from microgen.api.app import create_app
from microgen.profiling import DiagnosticEngine, Profiler
from microgen.scheduler import ContinuousBatchingScheduler
from microgen.runtime import KVCacheManager
from microgen.benchmarks import WorkloadGenerator
from microgen.devices import get_device
from microgen.backends.pytorch import PyTorchBackend
from microgen.scheduler.queue import Request


@click.group()
def cli() -> None:
    """MicroGen LLM Inference Server, SDK & Performance Suite CLI."""
    pass


def main() -> None:
    """CLI binary entry point."""
    cli()


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or local directory path.")
@click.option("--quantize", default=None, type=str, help="Weight quantization mode ('int8', 'fp8').")
@click.option("--tp-size", default=1, type=int, help="Tensor Parallelism world size (number of GPU ranks).")
@click.option("--device", default="cpu", help="Target hardware device ('cpu', 'cuda', 'cuda:0').")
@click.option("--max-tokens", default=128, type=int, help="Maximum new tokens per response.")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature.")
def chat(
    model: str,
    quantize: Optional[str],
    tp_size: int,
    device: str,
    max_tokens: int,
    temperature: float,
) -> None:
    """Launch an interactive, real-time streaming terminal chat session."""
    console = Console() if RICH_AVAILABLE else None

    if console:
        console.print(
            Panel(
                f"[bold cyan]MicroGen Interactive Terminal Chat[/bold cyan]\n"
                f"[dim]Model:[/dim] [green]{model}[/green] | "
                f"[dim]Device:[/dim] [yellow]{device}[/yellow] | "
                f"[dim]Quantization:[/dim] [magenta]{quantize or 'None'}[/magenta] | "
                f"[dim]TP Ranks:[/dim] [blue]{tp_size}[/blue]\n\n"
                f"[dim]Special Commands: /clear (clear screen), /stats (memory info), /exit (quit)[/dim]",
                title="⚡ MicroGen v1.0.0",
                expand=False,
            )
        )
        console.print("[dim]Loading model weights...[/dim]")
    else:
        click.echo(f"Starting MicroGen Chat session with model={model} on device={device}...")

    try:
        engine = LLMEngine.from_pretrained(
            model_name_or_path=model,
            quantize=quantize,
            tensor_parallel_size=tp_size,
            device=device,
        )
    except Exception as e:
        if console:
            console.print(f"[bold red]Failed to load engine:[/bold red] {e}")
        else:
            click.echo(f"Failed to load engine: {e}", err=True)
        sys.exit(1)

    if console:
        console.print("[bold green]Model loaded successfully! Type your message below.[/bold green]\n")
    else:
        click.echo("Model loaded. Type your message below.\n")

    while True:
        try:
            if console:
                prompt_text = Prompt.ask("[bold blue]User[/bold blue]").strip()
            else:
                prompt_text = input("User > ").strip()
        except (KeyboardInterrupt, EOFError):
            if console:
                console.print("\n[dim]Exiting chat session.[/dim]")
            else:
                click.echo("\nExiting chat session.")
            break

        if not prompt_text:
            continue

        cmd = prompt_text.lower()
        if cmd in ("/exit", "/quit"):
            if console:
                console.print("[dim]Exiting MicroGen chat. Goodbye![/dim]")
            else:
                click.echo("Exiting MicroGen chat. Goodbye!")
            break
        elif cmd == "/clear":
            click.clear()
            continue
        elif cmd == "/stats":
            mem = engine.get_memory_usage()
            if console:
                console.print(f"[bold cyan]Engine Stats:[/bold cyan] {mem}")
            else:
                click.echo(f"Engine Stats: {mem}")
            continue
        elif cmd == "/help":
            help_msg = "Commands: /exit (quit session), /clear (reset screen), /stats (show engine memory stats)"
            if console:
                console.print(f"[dim]{help_msg}[/dim]")
            else:
                click.echo(help_msg)
            continue

        if console:
            console.print("[bold green]MicroGen[/bold green] > ", end="")
            for token in engine.generate(prompt_text, max_new_tokens=max_tokens, stream=True, temperature=temperature):
                console.print(token, end="", highlight=False)
            console.print("\n")
        else:
            sys.stdout.write("MicroGen > ")
            for token in engine.generate(prompt_text, max_new_tokens=max_tokens, stream=True, temperature=temperature):
                sys.stdout.write(token)
                sys.stdout.flush()
            sys.stdout.write("\n\n")


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or local directory path.")
@click.option("--quantize", default=None, type=str, help="Weight quantization mode ('int8', 'fp8').")
@click.option("--tp-size", default=1, type=int, help="Tensor Parallelism world size.")
@click.option("--device", default="cpu", help="Target hardware device.")
@click.option("--host", default="0.0.0.0", help="HTTP server listen host.")
@click.option("--port", default=8000, type=int, help="HTTP server listen port.")
@click.option("--max-batch-size", default=8, type=int, help="Maximum continuous batching size.")
def serve(
    model: str,
    quantize: Optional[str],
    tp_size: int,
    device: str,
    host: str,
    port: int,
    max_batch_size: int,
) -> None:
    """Start OpenAI-compatible HTTP inference API server."""
    console = Console() if RICH_AVAILABLE else None

    if console:
        console.print(
            Panel(
                f"[bold cyan]MicroGen High-Performance Inference Server[/bold cyan]\n"
                f"[dim]Model:[/dim] [green]{model}[/green] | "
                f"[dim]Listen Address:[/dim] [yellow]http://{host}:{port}[/yellow]\n"
                f"[dim]Quantization:[/dim] [magenta]{quantize or 'None'}[/magenta] | "
                f"[dim]TP Ranks:[/dim] [blue]{tp_size}[/blue]",
                title="⚡ MicroGen Server",
                expand=False,
            )
        )
    else:
        click.echo(f"Initializing MicroGen server with model={model} on http://{host}:{port}")

    try:
        engine = LLMEngine.from_pretrained(
            model_name_or_path=model,
            quantize=quantize,
            tensor_parallel_size=tp_size,
            device=device,
        )
    except Exception as e:
        click.echo(f"Failed to initialize engine: {e}", err=True)
        sys.exit(1)

    kv_manager = KVCacheManager()
    scheduler = ContinuousBatchingScheduler(
        backend=engine.backend,
        kv_cache_manager=kv_manager,
        max_batch_size=max_batch_size,
        eos_token_id=engine.tokenizer.eos_token_id or 50256,
    )

    running = True

    def scheduler_loop():
        while running:
            scheduler.step()
            time.sleep(0.005)

    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    app = create_app(scheduler=scheduler, tokenizer=engine.tokenizer)
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        running = False


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or path.")
@click.option("--num-requests", default=5, type=int, help="Number of benchmark requests to generate.")
@click.option("--max-tokens", default=32, type=int, help="Maximum generated tokens per request.")
@click.option("--quantize", default=None, type=str, help="Weight quantization mode ('int8', 'fp8').")
@click.option("--tp-size", default=1, type=int, help="Tensor Parallelism world size.")
@click.option("--device", default="cpu", help="Target hardware device.")
def benchmark(
    model: str,
    num_requests: int,
    max_tokens: int,
    quantize: Optional[str],
    tp_size: int,
    device: str,
) -> None:
    """Run automated inference performance benchmark suite."""
    console = Console() if RICH_AVAILABLE else None

    if console:
        console.print(f"[bold cyan]Running MicroGen Benchmark Suite...[/bold cyan]")
        console.print(f"[dim]Model:[/dim] {model} | [dim]Requests:[/dim] {num_requests} | [dim]Device:[/dim] {device}")
    else:
        click.echo(f"Running benchmark on model={model} (num_requests={num_requests}, device={device})...")

    generator = WorkloadGenerator(tokenizer_name_or_path=model)
    suite = generator.generate_suite(
        name="cli_benchmark",
        num_requests=num_requests,
        target_len_range=(32, 64),
        max_new_tokens=max_tokens,
        seed=42,
    )

    engine = LLMEngine.from_pretrained(
        model_name_or_path=model,
        quantize=quantize,
        tensor_parallel_size=tp_size,
        device=device,
    )

    start_time = time.perf_counter()
    total_tokens_generated = 0
    latencies = []

    for req in suite.requests:
        t0 = time.perf_counter()
        output = engine.generate(req.prompt_text, max_new_tokens=max_tokens, stream=False)
        t1 = time.perf_counter()

        gen_tokens = len(engine.tokenizer.encode(output))
        total_tokens_generated += gen_tokens
        latencies.append(t1 - t0)

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000.0 if latencies else 0.0
    throughput = total_tokens_generated / max(total_time, 1e-5)

    if console:
        table = Table(title="⚡ MicroGen Benchmark Results")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        table.add_row("Model", model)
        table.add_row("Total Requests", str(num_requests))
        table.add_row("Total Generated Tokens", str(total_tokens_generated))
        table.add_row("Total Benchmark Duration", f"{total_time:.3f} s")
        table.add_row("Avg Request Latency", f"{avg_latency_ms:.2f} ms")
        table.add_row("System Throughput", f"{throughput:.2f} tokens/sec")

        console.print(table)
    else:
        click.echo("\n--- Benchmark Results ---")
        click.echo(f"Model:                   {model}")
        click.echo(f"Total Requests:          {num_requests}")
        click.echo(f"Total Generated Tokens:  {total_tokens_generated}")
        click.echo(f"Total Time:              {total_time:.3f} s")
        click.echo(f"Avg Latency:             {avg_latency_ms:.2f} ms")
        click.echo(f"System Throughput:       {throughput:.2f} tokens/sec")


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or path.")
@click.option("--prompt", required=True, help="Input prompt text for generation.")
@click.option("--max-tokens", default=32, type=int, help="Maximum new tokens to generate.")
@click.option("--quantize", default=None, type=str, help="Weight quantization mode ('int8', 'fp8').")
@click.option("--tp-size", default=1, type=int, help="Tensor Parallelism world size.")
@click.option("--device", default="cpu", help="Target hardware device.")
@click.option("--stream/--no-stream", default=False, help="Enable streaming token generation.")
def generate(
    model: str,
    prompt: str,
    max_tokens: int,
    quantize: Optional[str],
    tp_size: int,
    device: str,
    stream: bool,
) -> None:
    """Run standalone text generation for a single prompt."""
    click.echo(f"Loading model {model} on {device}...")
    engine = LLMEngine.from_pretrained(
        model_name_or_path=model,
        quantize=quantize,
        tensor_parallel_size=tp_size,
        device=device,
    )

    if stream:
        click.echo("\n--- Generated Output (Streaming) ---")
        for token in engine.generate(prompt, max_new_tokens=max_tokens, stream=True):
            sys.stdout.write(token)
            sys.stdout.flush()
        sys.stdout.write("\n")
    else:
        output = engine.generate(prompt, max_new_tokens=max_tokens, stream=False)
        click.echo("\n--- Generated Output ---")
        click.echo(output)


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or path.")
@click.option("--prompt", default="Hello microgen benchmark", help="Sample prompt for profiling.")
@click.option("--device", default="cpu", help="Target hardware device (cpu, cuda).")
def profile(model: str, prompt: str, device: str) -> None:
    """Profile inference execution and output performance diagnostic report."""
    click.echo(f"Profiling model {model} on {device}...")
    dev = get_device(device)
    backend = PyTorchBackend(device=dev)
    backend.load_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    profiler = Profiler()
    with profiler.profile("prefill"):
        input_tensor = dev.to_device(torch.tensor([[100, 200, 300]], dtype=torch.long))
        logits, cache = backend.prefill(input_ids=input_tensor)

    for _ in range(5):
        with profiler.profile("decode"):
            next_token = backend.sample(logits)
            token_tensor = dev.to_device(next_token.unsqueeze(0))
            logits, cache = backend.decode(token_ids=token_tensor, cache=cache)

    diagnostics = DiagnosticEngine()
    report = diagnostics.analyze(profiler)

    click.echo("\n=== MicroGen Diagnostic Report ===")
    click.echo(f"Primary Bottleneck: {report.primary_bottleneck}")
    click.echo(f"Prefill/Decode Ratio: {report.prefill_decode_ratio}")
    click.echo("Recommendations:")
    for rec in report.recommendations:
        click.echo(f" - {rec}")
    click.echo("\nMetrics Breakdown:")
    click.echo(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    cli()
