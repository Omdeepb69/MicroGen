"""Unit tests for MicroGen CLI commands."""

import pytest
from click.testing import CliRunner
from microgen.cli.main import cli

TINY_MODEL = "sshleifer/tiny-gpt2"


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MicroGen LLM Inference Server" in result.output


def test_cli_generate_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate", "--model", TINY_MODEL, "--prompt", "Hello world", "--max-tokens", "5", "--device", "cpu"],
    )
    assert result.exit_code == 0
    assert "Generated Output" in result.output


def test_cli_generate_stream_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["generate", "--model", TINY_MODEL, "--prompt", "Hello world", "--max-tokens", "5", "--device", "cpu", "--stream"],
    )
    assert result.exit_code == 0
    assert "Generated Output (Streaming)" in result.output


def test_cli_benchmark_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["benchmark", "--model", TINY_MODEL, "--num-requests", "2", "--max-tokens", "4", "--device", "cpu"],
    )
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output or "MicroGen Benchmark Results" in result.output


def test_cli_profile_command():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["profile", "--model", TINY_MODEL, "--prompt", "Hello test", "--device", "cpu"],
    )
    assert result.exit_code == 0
    assert "MicroGen Diagnostic Report" in result.output


def test_cli_chat_command_exit():
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["chat", "--model", TINY_MODEL, "--max-tokens", "5", "--device", "cpu"],
        input="/exit\n",
    )
    assert result.exit_code == 0
