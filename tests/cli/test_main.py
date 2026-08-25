from diflow.cli import main as cli_main


def test_serve_subcommand_delegates_to_shared_implementation(monkeypatch):
    received = {}

    def fake_serve_main(arguments, prog):
        received["arguments"] = arguments
        received["prog"] = prog
        return 7

    monkeypatch.setattr(cli_main, "serve_main", fake_serve_main)

    assert cli_main.main(["serve", "--workflow", "flux-schnell"]) == 7
    assert received == {
        "arguments": ["--workflow", "flux-schnell"],
        "prog": "diflow serve",
    }
