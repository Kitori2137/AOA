from types import SimpleNamespace

from AOA import cli


def test_interactive_generate(monkeypatch):
    answers = iter(["800", "1", "0.2", "42", "1", "48", "1", "72", "kwadrat,trojkat,trapez", "bawelna,mikrofibra"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    called = {"ok": False}

    def fake_generate(args):
        called["ok"] = True
        assert args.n == 800
        assert args.machines == 1
        assert args.test_size == 0.2
        assert args.seed == 42
        return 0

    monkeypatch.setattr(cli, "command_generate", fake_generate)
    code = cli.interactive_generate()

    assert code == 0
    assert called["ok"] is True


def test_interactive_train(monkeypatch):
    answers = iter(["data/train.csv", "Quality,Delay", "classic", "0.8"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_train(args):
        called["ok"] = True
        assert args.data == "data/train.csv"
        assert args.models == "Quality,Delay"
        assert args.backend == "classic"
        assert args.train_ratio == 0.8
        return 0

    monkeypatch.setattr(cli, "command_train", fake_train)
    code = cli.interactive_train()

    assert code == 0
    assert called["ok"] is True


def test_interactive_solve(monkeypatch):
    answers = iter(["models/model.pkl", "data/test.csv"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_solve(args):
        called["ok"] = True
        assert args.model == "models/model.pkl"
        assert args.data == "data/test.csv"
        return 0

    monkeypatch.setattr(cli, "command_solve", fake_solve)
    code = cli.interactive_solve()

    assert code == 0
    assert called["ok"] is True


def test_interactive_sto_run(monkeypatch):
    answers = iter(["Z1,Z2,Z3", "10,20,100", "150,30,110", "MT,MO"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_sto(args):
        called["ok"] = True
        assert args.jobs == "Z1,Z2,Z3"
        assert args.times == "10,20,100"
        assert args.deadlines == "150,30,110"
        assert args.methods == "MT,MO"
        return 0

    monkeypatch.setattr(cli, "command_sto_run", fake_sto)
    code = cli.interactive_sto_run()

    assert code == 0
    assert called["ok"] is True


def test_interactive_summary(monkeypatch):
    answers = iter([
        "Quality,Delay",
        "tabpfn",
        "800",
        "1",
        "0.2",
        "42",
        "1",
        "48",
        "1",
        "72",
        "kwadrat,trojkat,trapez",
        "bawelna,mikrofibra,poliester,wiskoza",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_summary(args):
        called["ok"] = True
        assert args.models == "Quality,Delay"
        assert args.backend == "tabpfn"
        assert args.n == 800
        return 0

    monkeypatch.setattr(cli, "command_summary", fake_summary)
    code = cli.interactive_summary()

    assert code == 0
    assert called["ok"] is True


def test_interactive_status(monkeypatch):
    answers = iter(["data/train.csv", "0.8"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_status(args):
        called["ok"] = True
        assert args.data == "data/train.csv"
        assert args.train_ratio == 0.8
        return 0

    monkeypatch.setattr(cli, "command_status", fake_status)
    code = cli.interactive_status()

    assert code == 0
    assert called["ok"] is True


def test_interactive_workflow(monkeypatch):
    answers = iter([
        "800",
        "1",
        "0.2",
        "42",
        "1",
        "48",
        "1",
        "72",
        "kwadrat,trojkat,trapez",
        "bawelna,mikrofibra,poliester,wiskoza",
        "Quality",
        "classic",
        "nie",
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    called = {"ok": False}

    def fake_workflow(args):
        called["ok"] = True
        assert args.n == 800
        assert args.models == "Quality"
        assert args.backend == "classic"
        assert args.skip_solve is False
        return 0

    monkeypatch.setattr(cli, "command_workflow", fake_workflow)
    code = cli.interactive_workflow()

    assert code == 0
    assert called["ok"] is True


def test_command_interactive_exit(monkeypatch, capsys):
    answers = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    code = cli.command_interactive(SimpleNamespace(quick=False))
    assert code == 0

    out = capsys.readouterr().out
    assert "TRYB INTERAKTYWNY" in out
    assert "Koniec." in out


def test_command_interactive_invalid_choice_then_exit(monkeypatch, capsys):
    answers = iter(["99", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    code = cli.command_interactive(SimpleNamespace(quick=False))
    assert code == 0

    out = capsys.readouterr().out
    assert "Nieprawidłowy wybór." in out
    assert "Koniec." in out
