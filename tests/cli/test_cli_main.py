import pytest

from AOA import cli


def test_main_help_exits_with_code_0(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "aoa-cli" in out
    assert "generate" in out
    assert "train" in out
    assert "workflow" in out


def test_subcommand_help_exits_with_code_0(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--data" in out
    assert "--models" in out
    assert "--backend" in out


def test_main_returns_1_for_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["unknown-command"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err or "nieprawidłowy" in err.lower()


def test_main_returns_130_on_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(cli, "command_generate", lambda args: (_ for _ in ()).throw(KeyboardInterrupt()))

    code = cli.main(["generate"])
    assert code == 130


def test_main_returns_1_and_logs_on_exception(monkeypatch, capsys):
    monkeypatch.setattr(cli, "command_generate", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(cli, "write_exception_log", lambda *_args, **_kwargs: "fake_log_path.txt")

    code = cli.main(["generate"])
    assert code == 1

    err = capsys.readouterr().err
    assert "RuntimeError" in err
    assert "boom" in err
    assert "fake_log_path.txt" in err


def test_parse_csv_list_handles_none_and_values():
    assert cli.parse_csv_list(None) == []
    assert cli.parse_csv_list("") == []
    assert cli.parse_csv_list("A,B,C") == ["A", "B", "C"]
    assert cli.parse_csv_list(" A , B ,  C ") == ["A", "B", "C"]


def test_validate_models_accepts_valid_values():
    result = cli.validate_models(["MT", "MO"], {"MT", "MO", "MZO"}, "metody STO")
    assert result == ["MT", "MO"]


def test_validate_models_rejects_invalid_values():
    with pytest.raises(ValueError, match="Nieprawidłowe metody STO"):
        cli.validate_models(["MT", "XYZ"], {"MT", "MO", "MZO"}, "metody STO")
