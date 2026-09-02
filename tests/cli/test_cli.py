"""Integration unit tests for microgen Click CLI commands."""

from click.testing import CliRunner
import pytest
from microgen.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "MicroGen LLM Inference Server" in result.output


def test_cli_generate_command(runner):
    result = runner.invoke(
        cli,
        [
            "generate",
            "--model",
            "sshleifer/tiny-gpt2",
            "--prompt",
            "Hello world",
            "--max-tokens",
            "5",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert "Generated Output" in result.output


def test_cli_profile_command(runner):
    result = runner.invoke(
        cli,
        [
            "profile",
            "--model",
            "sshleifer/tiny-gpt2",
            "--prompt",
            "Test profile prompt",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert "MicroGen Diagnostic Report" in result.output
    assert "Primary Bottleneck" in result.output
