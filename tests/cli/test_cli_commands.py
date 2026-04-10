from pathlib import Path
from types import SimpleNamespace

import pytest

from AOA import cli


def norm(text: str) -> str:
    return text.replace("\\", "/")


def test_command_generate_prints_paths(monkeypatch, capsys):
    fake_result = {
        "full_path": Path("data/full.csv"),
        "train_path": Path("data/train.csv"),
        "test_path": Path("data/test.csv"),
        "messages": ["msg1", "msg2"],
    }

    monkeypatch.setattr(cli, "generate_and_store_datasets", lambda **kwargs: fake_result)

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
    )

    code = cli.command_generate(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "GENEROWANIE" in out
    assert "FULL: data/full.csv" in out
    assert "TRAIN: data/train.csv" in out
    assert "TEST: data/test.csv" in out


def test_command_train_runs_ml_flow(monkeypatch, capsys, tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "resolve_existing_file", lambda path, label: data_file)
    monkeypatch.setattr(
        cli,
        "load_training_data",
        lambda path, train_ratio: {
            "full_df": [1, 2, 3, 4],
            "train_df": ["train-data"],
            "test_df": ["test-data"],
            "messages": ["loaded"],
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

    args = SimpleNamespace(
        data=str(data_file),
        models="Quality,Delay",
        backend="classic",
        train_ratio=0.8,
        n_meta=None,
        machines_meta=None,
        shapes_meta="",
        materials_meta="",
    )

    code = cli.command_train(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "WCZYTANIE DANYCH" in out
    assert "TRENING ML" in out
    assert "MODEL_ML: models/model_ml.pkl" in out


def test_command_train_runs_sto_flow(monkeypatch, capsys, tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "resolve_existing_file", lambda path, label: data_file)
    monkeypatch.setattr(
        cli,
        "load_training_data",
        lambda path, train_ratio: {
            "full_df": [1, 2, 3],
            "train_df": ["train-data"],
            "test_df": ["test-data"],
            "messages": ["loaded"],
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
        data=str(data_file),
        models="MT,MO",
        backend="classic",
        train_ratio=0.8,
        n_meta=None,
        machines_meta=None,
        shapes_meta="",
        materials_meta="",
    )

    code = cli.command_train(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "ZAPIS STO" in out
    assert "MODEL_STO: models/model_sto.pkl" in out


def test_command_train_rejects_invalid_model(monkeypatch, tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cli, "resolve_existing_file", lambda path, label: data_file)

    args = SimpleNamespace(
        data=str(data_file),
        models="Quality,BADMODEL",
        backend="classic",
        train_ratio=0.8,
        n_meta=None,
        machines_meta=None,
        shapes_meta="",
        materials_meta="",
    )

    with pytest.raises(ValueError, match="Nieprawidłowe modele"):
        cli.command_train(args)


def test_command_solve_prints_result_path(monkeypatch, capsys, tmp_path):
    model_file = tmp_path / "model.pkl"
    data_file = tmp_path / "data.csv"
    model_file.write_text("x", encoding="utf-8")
    data_file.write_text("x", encoding="utf-8")

    def fake_resolve(path, label):
        return model_file if label == "pliku modelu" else data_file

    monkeypatch.setattr(cli, "resolve_existing_file", fake_resolve)
    monkeypatch.setattr(
        cli,
        "solve_models_flow",
        lambda model_path, data_path: {
            "result_path": Path("data/wynik.csv"),
            "messages": ["done"],
        },
    )

    args = SimpleNamespace(model=str(model_file), data=str(data_file))
    code = cli.command_solve(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "ROZWIĄZYWANIE ML" in out
    assert "WYNIK: data/wynik.csv" in out


def test_command_sto_run_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "analyze_sto_models",
        lambda **kwargs: {
            "report": "RAPORT STO TEST",
            "saved_paths": [{"method": "MT", "path": Path("data/sto_mt.csv"), "sto": 12.5}],
            "best_path": Path("data/best.csv"),
            "messages": ["sto done"],
        },
    )

    args = SimpleNamespace(
        jobs="Z1,Z2,Z3",
        times="10,20,100",
        deadlines="150,30,110",
        methods="MT,MO",
    )

    code = cli.command_sto_run(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "RAPORT STO" in out
    assert "RAPORT STO TEST" in out
    assert "data/sto_mt.csv" in out
    assert "BEST: data/best.csv" in out


def test_command_sto_train_prints_saved_model(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "train_sto_models_flow",
        lambda methods: {
            "model_path": Path("models/sto.pkl"),
            "messages": ["saved"],
        },
    )

    args = SimpleNamespace(methods="MT,MO,MZO")
    code = cli.command_sto_train(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "ZAPIS STO" in out
    assert "MODEL_STO: models/sto.pkl" in out


def test_command_sto_solve_prints_report(monkeypatch, capsys, tmp_path):
    model_file = tmp_path / "sto.pkl"
    data_file = tmp_path / "data.csv"
    model_file.write_text("x", encoding="utf-8")
    data_file.write_text("x", encoding="utf-8")

    def fake_resolve(path, label):
        return model_file if "modelu STO" in label else data_file

    monkeypatch.setattr(cli, "resolve_existing_file", fake_resolve)
    monkeypatch.setattr(
        cli,
        "solve_sto_with_saved_model",
        lambda model_path, data_path: {
            "report": "STO REPORT",
            "saved_paths": [{"method": "MO", "path": Path("data/mo.csv"), "sto": 8.0}],
            "best_path": Path("data/best.csv"),
            "messages": ["ok"],
        },
    )

    args = SimpleNamespace(model=str(model_file), data=str(data_file))
    code = cli.command_sto_solve(args)
    assert code == 0

    out = norm(capsys.readouterr().out)
    assert "STO REPORT" in out
    assert "data/mo.csv" in out
    assert "BEST: data/best.csv" in out


def test_command_preview_prints_dataframe_preview(monkeypatch, capsys, tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "resolve_existing_file", lambda path, label: data_file)
    monkeypatch.setattr(cli, "load_csv", lambda path: {"dummy": "df"})
    monkeypatch.setattr(
        cli,
        "build_dataframe_preview_text",
        lambda df, title, max_rows: f"{title} | rows={max_rows}",
    )

    args = SimpleNamespace(data=str(data_file), rows=10)
    code = cli.command_preview(args)
    assert code == 0

    out = capsys.readouterr().out
    assert "Podgląd: data.csv | rows=10" in out


def test_command_summary_prints_summary(capsys):
    args = SimpleNamespace(
        models="Quality,Delay,MT",
        backend="tabpfn",
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
    )

    code = cli.command_summary(args)
    assert code == 0

    out = capsys.readouterr().out
    assert "PODSUMOWANIE KONFIGURACJI" in out
    assert "Backend ML: TabPFN" in out or "Backend ML: TabPFN (eksperymentalny)" in out
    assert "Quality" in out
    assert "Delay" in out
    assert "MT" in out


def test_command_status_without_file(capsys):
    args = SimpleNamespace(data="", train_ratio=0.8)
    code = cli.command_status(args)
    assert code == 0

    out = capsys.readouterr().out
    assert "STATUS" in out
    assert "Brak danych treningowych" in out


def test_command_status_with_file(monkeypatch, capsys, tmp_path):
    data_file = tmp_path / "data.csv"
    data_file.write_text("x", encoding="utf-8")

    monkeypatch.setattr(cli, "resolve_existing_file", lambda path, label: data_file)
    monkeypatch.setattr(
        cli,
        "load_training_data",
        lambda path, train_ratio: {
            "train_df": [1, 2, 3],
            "test_df": [4],
        },
    )

    args = SimpleNamespace(data=str(data_file), train_ratio=0.8)
    code = cli.command_status(args)
    assert code == 0

    out = capsys.readouterr().out
    assert "STATUS" in out
    assert "Train: 3 rekordów" in out
    assert "Test: 1 rekordów" in out
