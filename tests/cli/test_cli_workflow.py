from pathlib import Path
from types import SimpleNamespace

from AOA import cli


def norm(text: str) -> str:
    return text.replace("\\", "/")


def test_command_workflow_runs_generate_train_and_solve(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "generate_and_store_datasets",
        lambda **kwargs: {
            "full_df": ["full"],
            "train_df": ["train"],
            "test_df": ["test"],
            "full_path": Path("data/full.csv"),
            "train_path": Path("data/train.csv"),
            "test_path": Path("data/test.csv"),
            "messages": ["generated"],
        },
    )

    monkeypatch.setattr(
        cli,
        "train_models_flow",
        lambda **kwargs: {
            "model_path": Path("models/model_ml.pkl"),
            "messages": ["trained"],
        },
    )

    monkeypatch.setattr(
        cli,
        "solve_models_flow",
        lambda model_path, data_path: {
            "result_path": Path("data/result.csv"),
            "messages": ["solved"],
        },
    )

    monkeypatch.setattr(
        cli,
        "train_sto_models_flow",
        lambda methods: {
            "model_path": Path("models/model_sto.pkl"),
            "messages": ["sto saved"],
        },
    )

    args = SimpleNamespace(
        n=800,
        machines=1,
        test_size=0.2,
        seed=42,
        prod_min=1.0,
        prod_max=48.0,
        deadline_min=1.0,
        deadline_max=72.0,
        shapes="kwadrat,trojkat,trapez",
        materials="bawelna,mikrofibra,poliester,wiskoza",
        models="Quality,Delay,MT,MO",
        backend="classic",
        skip_solve=False,
    )

    code = cli.command_workflow(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "WORKFLOW | GENEROWANIE" in out
    assert "WORKFLOW | TRENING ML" in out
    assert "WORKFLOW | ZAPIS STO" in out
    assert "WORKFLOW | ROZWIĄZYWANIE TESTU" in out
    assert "WYNIK_TEST: data/result.csv" in out
    assert "MODEL_ML: models/model_ml.pkl" in out


def test_command_workflow_skips_solve(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "generate_and_store_datasets",
        lambda **kwargs: {
            "full_df": ["full"],
            "train_df": ["train"],
            "test_df": ["test"],
            "full_path": Path("data/full.csv"),
            "train_path": Path("data/train.csv"),
            "test_path": Path("data/test.csv"),
            "messages": ["generated"],
        },
    )

    monkeypatch.setattr(
        cli,
        "train_models_flow",
        lambda **kwargs: {
            "model_path": Path("models/model_ml.pkl"),
            "messages": ["trained"],
        },
    )

    called = {"solve": False}

    def fake_solve(*args, **kwargs):
        called["solve"] = True
        return {"result_path": Path("data/result.csv"), "messages": ["solved"]}

    monkeypatch.setattr(cli, "solve_models_flow", fake_solve)

    args = SimpleNamespace(
        n=800,
        machines=1,
        test_size=0.2,
        seed=42,
        prod_min=1.0,
        prod_max=48.0,
        deadline_min=1.0,
        deadline_max=72.0,
        shapes="kwadrat,trojkat,trapez",
        materials="bawelna,mikrofibra,poliester,wiskoza",
        models="Quality",
        backend="classic",
        skip_solve=True,
    )

    code = cli.command_workflow(args)
    assert code == 0
    assert called["solve"] is False

    out = norm(capsys.readouterr().out)
    assert "WORKFLOW | KONIEC" in out
    assert "MODEL_ML: models/model_ml.pkl" in out


def test_command_workflow_with_only_sto(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "generate_and_store_datasets",
        lambda **kwargs: {
            "full_df": ["full"],
            "train_df": ["train"],
            "test_df": ["test"],
            "full_path": Path("data/full.csv"),
            "train_path": Path("data/train.csv"),
            "test_path": Path("data/test.csv"),
            "messages": ["generated"],
        },
    )

    monkeypatch.setattr(
        cli,
        "train_sto_models_flow",
        lambda methods: {
            "model_path": Path("models/model_sto.pkl"),
            "messages": ["sto saved"],
        },
    )

    args = SimpleNamespace(
        n=800,
        machines=1,
        test_size=0.2,
        seed=42,
        prod_min=1.0,
        prod_max=48.0,
        deadline_min=1.0,
        deadline_max=72.0,
        shapes="kwadrat,trojkat,trapez",
        materials="bawelna,mikrofibra,poliester,wiskoza",
        models="MT,MO",
        backend="classic",
        skip_solve=False,
    )

    code = cli.command_workflow(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "WORKFLOW | ZAPIS STO" in out
    assert "WORKFLOW | KONIEC" in out
    assert "FULL: data/full.csv" in out


def test_command_workflow_rejects_invalid_models():
    args = SimpleNamespace(
        n=800,
        machines=1,
        test_size=0.2,
        seed=42,
        prod_min=1.0,
        prod_max=48.0,
        deadline_min=1.0,
        deadline_max=72.0,
        shapes="kwadrat,trojkat,trapez",
        materials="bawelna,mikrofibra,poliester,wiskoza",
        models="Quality,BADMODEL",
        backend="classic",
        skip_solve=False,
    )

    try:
        cli.command_workflow(args)
        assert False, "Expected ValueError for invalid model"
    except ValueError as e:
        assert "Nieprawidłowe modele" in str(e)
