"""Unified Click CLI entry point for MicroGen serving, standalone generation, and profiling."""

import json
import threading
import time
import click
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

from microgen.api.app import create_app
from microgen.backends.pytorch import PyTorchBackend
from microgen.devices import get_device
from microgen.profiling import DiagnosticEngine, Profiler
from microgen.runtime import KVCacheManager
from microgen.scheduler import ContinuousBatchingScheduler
from microgen.scheduler.queue import Request


@click.group()
def cli() -> None:
    """MicroGen LLM Inference Server & Diagnostic Engine CLI."""
    pass


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or local path.")
@click.option("--device", default="cpu", help="Target hardware device (cpu, cuda, cuda:0).")
@click.option("--host", default="0.0.0.0", help="HTTP server listen host.")
@click.option("--port", default=8000, type=int, help="HTTP server listen port.")
@click.option("--max-batch-size", default=8, type=int, help="Maximum continuous batching size.")
def serve(model: str, device: str, host: str, port: int, max_batch_size: int) -> None:
    """Start OpenAI-compatible HTTP inference API server."""
    click.echo(f"Initializing MicroGen server with model={model} on device={device}...")
    dev = get_device(device)
    model_inst = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    backend = PyTorchBackend(device=dev)
    backend.load_model(model, model_instance=model_inst)
    kv_manager = KVCacheManager()

    scheduler = ContinuousBatchingScheduler(
        backend=backend,
        kv_cache_manager=kv_manager,
        max_batch_size=max_batch_size,
        eos_token_id=tokenizer.eos_token_id or 50256,
    )

    # Background scheduler iteration thread
    running = True

    def scheduler_loop():
        while running:
            scheduler.step()
            time.sleep(0.005)

    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()

    app = create_app(scheduler=scheduler, tokenizer=tokenizer)
    click.echo(f"Starting MicroGen HTTP server on http://{host}:{port}")
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        running = False


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or path.")
@click.option("--prompt", required=True, help="Input prompt text for generation.")
@click.option("--max-tokens", default=32, type=int, help="Maximum new tokens to generate.")
@click.option("--device", default="cpu", help="Target hardware device (cpu, cuda).")
def generate(model: str, prompt: str, max_tokens: int, device: str) -> None:
    """Run standalone text generation for a single prompt."""
    click.echo(f"Loading model {model} on {device}...")
    dev = get_device(device)
    model_inst = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    backend = PyTorchBackend(device=dev)
    backend.load_model(model, model_instance=model_inst)
    kv_manager = KVCacheManager()

    scheduler = ContinuousBatchingScheduler(
        backend=backend,
        kv_cache_manager=kv_manager,
        eos_token_id=tokenizer.eos_token_id or 50256,
    )

    token_ids = tokenizer.encode(prompt)
    req = Request(
        request_id="cli-req-1",
        prompt=prompt,
        prompt_ids=token_ids,
        max_new_tokens=max_tokens,
    )
    scheduler.add_request(req)

    scheduler.run_until_complete()
    generated_text = tokenizer.decode(req.generated_token_ids)
    click.echo("\n--- Generated Output ---")
    click.echo(generated_text)


@cli.command()
@click.option("--model", default="sshleifer/tiny-gpt2", help="HuggingFace model ID or path.")
@click.option("--prompt", default="Hello microgen benchmark", help="Sample prompt for profiling.")
@click.option("--device", default="cpu", help="Target hardware device (cpu, cuda).")
def profile(model: str, prompt: str, device: str) -> None:
    """Profile inference execution and output performance diagnostic report."""
    click.echo(f"Profiling model {model} on {device}...")
    dev = get_device(device)
    model_inst = AutoModelForCausalLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)

    backend = PyTorchBackend(device=dev)
    backend.load_model(model, model_instance=model_inst)

    profiler = Profiler()
    token_ids = tokenizer.encode(prompt)

    with profiler.profile("prefill"):
        input_tensor = dev.to_device(tokenizer.encode(prompt, return_tensors="pt"))
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
