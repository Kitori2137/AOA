from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from AOA.config import DATA_DIR, MODELS_DIR
from AOA.core.data_generation import generate_production_data
from AOA.core.data_io import load_csv, save_csv
from AOA.core.dataset_ops import split_train_test
from AOA.core.evaluation import (
    append_metrics_row,
    calculate_classification_metrics,
    calculate_regression_metrics,
    fill_missing_values,
    transform_numeric_columns,
)
from AOA.core.features import prepare_features
from AOA.core.models import load_model_pack, save_model_pack, train_selected_models
from AOA.core.scheduling import simulate_schedule
from AOA.core.sto_models import (
    apply_sto_result_to_dataframe,
    build_sto_report,
    dataframe_to_jobs,
    parse_jobs,
    run_selected_sto_models,
)

POSITIVE_VALUES_MESSAGE = "Głuptasie, czemu wpisujesz ujemne rzeczy.\nWpisuj dodatnie"

ML_MODEL_NAMES = {"Quality", "Delay", "Schedule"}
STO_MODEL_NAMES = {"MT", "MO", "MZO", "GENETIC"}


def _parse_positive_int(value, field_name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Pole '{field_name}' musi być liczbą całkowitą.")

    if number <= 0:
        raise ValueError(POSITIVE_VALUES_MESSAGE)
    return number


def _parse_positive_float(value, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Pole '{field_name}' musi być liczbą.")

    if number <= 0:
        raise ValueError(POSITIVE_VALUES_MESSAGE)
    return number


def split_selected_models(selected_models: list[str]) -> tuple[list[str], list[str]]:
    selected_models = selected_models or []
    ml_models = [name for name in selected_models if name in ML_MODEL_NAMES]
    sto_models = [name for name in selected_models if name in STO_MODEL_NAMES]
    return ml_models, sto_models


def parse_generation_config(raw_config: dict) -> dict:
    n = _parse_positive_int(raw_config.get("n", ""), "n")
    n_machines = _parse_positive_int(raw_config.get("n_machines", ""), "n_machines")

    test_size = float(raw_config.get("test_size", ""))
    if test_size <= 0:
        raise ValueError(POSITIVE_VALUES_MESSAGE)
    if test_size >= 1:
        raise ValueError("Test size musi być mniejsze od 1.")

    seed = _parse_positive_int(raw_config.get("seed", ""), "seed")
    prod_min = _parse_positive_float(raw_config.get("prod_min", ""), "prod_min")
    prod_max = _parse_positive_float(raw_config.get("prod_max", ""), "prod_max")
    deadline_min = _parse_positive_float(raw_config.get("deadline_min", ""), "deadline_min")
    deadline_max = _parse_positive_float(raw_config.get("deadline_max", ""), "deadline_max")

    if prod_min > prod_max:
        raise ValueError("Minimalny czas produkcji nie może być większy niż maksymalny.")
    if deadline_min > deadline_max:
        raise ValueError("Minimalny bufor terminu nie może być większy niż maksymalny.")

    selected_ksztalty = raw_config.get("selected_ksztalty", [])
    selected_materialy = raw_config.get("selected_materialy", [])

    if not selected_ksztalty:
        raise ValueError("Wybierz przynajmniej jeden kształt.")
    if not selected_materialy:
        raise ValueError("Wybierz przynajmniej jeden materiał.")

    return {
        "n": n,
        "n_machines": n_machines,
        "test_size": test_size,
        "seed": seed,
        "prod_min": prod_min,
        "prod_max": prod_max,
        "deadline_min": deadline_min,
        "deadline_max": deadline_max,
        "selected_ksztalty": selected_ksztalty,
        "selected_materialy": selected_materialy,
    }


def build_main_page_summary(config: dict) -> str:
    selected_models = config.get("selected_models", [])
    ml_models, sto_models = split_selected_models(selected_models)

    selected_ksztalty = config.get("selected_ksztalty", [])
    selected_materialy = config.get("selected_materialy", [])
    backend = config.get("backend", "classic")
    backend_label = "TabPFN (eksperymentalny)" if backend == "tabpfn" else "Klasyczny"

    return (
        "AKTUALNA KONFIGURACJA\n"
        "======================\n\n"
        f"Backend ML: {backend_label}\n\n"
        f"Modele ML:\n - {', '.join(ml_models) if ml_models else 'brak'}\n\n"
        f"Modele STO:\n - {', '.join(sto_models) if sto_models else 'brak'}\n\n"
        f"Liczba rekordów: {config.get('n', '')}\n"
        f"Liczba maszyn: {config.get('n_machines', '')}\n"
        f"Test size: {config.get('test_size', '')}\n"
        f"Seed: {config.get('seed', '')}\n\n"
        f"Czas produkcji [h]: {config.get('prod_min', '')} -> {config.get('prod_max', '')}\n"
        f"Bufor terminu [h]: {config.get('deadline_min', '')} -> {config.get('deadline_max', '')}\n\n"
        f"Kształty:\n - {', '.join(selected_ksztalty) if selected_ksztalty else 'brak'}\n\n"
        f"Materiały:\n - {', '.join(selected_materialy) if selected_materialy else 'brak'}\n"
    )


def build_main_page_status(df_train=None, df_test=None) -> str:
    if df_train is None and df_test is None:
        return "Brak danych treningowych"

    train_count = len(df_train) if df_train is not None else 0
    test_count = len(df_test) if df_test is not None else 0
    total_count = train_count + test_count

    return (
        "Aktualnie załadowane dane\n"
        f"Łącznie: {total_count} rekordów\n"
        f"Train: {train_count} rekordów\n"
        f"Test: {test_count} rekordów"
    )


def build_dataframe_preview_text(df, title="Podgląd danych", max_rows=15):
    if df is None:
        return f"{title}\n\nBrak danych."
    if df.empty:
        return f"{title}\n\nDataFrame jest pusty."

    preview = df.head(max_rows).to_string(index=True)
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"Liczba rekordów: {len(df)}\n"
        f"Liczba kolumn: {len(df.columns)}\n\n"
        f"{preview}"
    )


def generate_and_store_datasets_from_config(raw_config: dict):
    parsed = parse_generation_config(raw_config)
    return generate_and_store_datasets(
        n=parsed["n"],
        n_machines=parsed["n_machines"],
        test_size=parsed["test_size"],
        seed=parsed["seed"],
        ksztalty=parsed["selected_ksztalty"],
        materialy=parsed["selected_materialy"],
        production_time_range=(parsed["prod_min"], parsed["prod_max"]),
        deadline_buffer_range=(parsed["deadline_min"], parsed["deadline_max"]),
    )


def generate_and_store_datasets(
    n=5000,
    n_machines=1,
    test_size=0.2,
    seed=42,
    ksztalty=None,
    materialy=None,
    production_time_range=(1, 48),
    deadline_buffer_range=(1, 72),
):
    full_df, train_df, test_df = generate_production_data(
        n=n,
        n_machines=n_machines,
        test_size=test_size,
        seed=seed,
        ksztalty=ksztalty,
        materialy=materialy,
        production_time_range=production_time_range,
        deadline_buffer_range=deadline_buffer_range,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = DATA_DIR / f"dane_{stamp}.csv"
    train_path = DATA_DIR / f"train_{stamp}.csv"
    test_path = DATA_DIR / f"test_{stamp}.csv"

    save_csv(full_df, full_path)
    save_csv(train_df, train_path)
    save_csv(test_df, test_path)

    return {
        "full_df": full_df,
        "train_df": train_df,
        "test_df": test_df,
        "full_path": full_path,
        "train_path": train_path,
        "test_path": test_path,
        "messages": [
            f"✔ Wygenerowano pełny zbiór: {full_path}",
            f"✔ Wygenerowano train: {train_path}",
            f"✔ Wygenerowano test: {test_path}",
        ],
    }


def load_and_prepare_visual_file(path, train_ratio=0.8):
    df = load_csv(path)
    columns = list(df.columns)

    if not columns:
        raise ValueError("Plik CSV nie zawiera kolumn")

    x_default = columns[0]
    y_default = columns[1] if len(columns) > 1 else columns[0]

    return {
        "df": df,
        "columns": columns,
        "x_default": x_default,
        "y_default": y_default,
    }


def prepare_results_analysis(
    df,
    selected_cols=None,
    transformation=None,
    target=None,
    mode=None,
):
    if df is None or df.empty:
        raise ValueError("Brak danych do analizy")

    if selected_cols is not None or target is not None or mode is not None:
        if not selected_cols:
            raise ValueError("Nie wybrano kolumn do analizy")

        df_to_show = df[selected_cols].copy()
        df_to_show = fill_missing_values(df_to_show)
        df_to_show = transform_numeric_columns(df_to_show, transformation or "Surowe")

        if target not in df_to_show.columns:
            raise ValueError("Nieprawidłowy target")

        if mode == "regresja":
            metrics = calculate_regression_metrics(df_to_show, target)
        elif mode == "klasyfikacja":
            metrics = calculate_classification_metrics(df_to_show, target)
        else:
            raise ValueError("Nieznany tryb analizy")

        df_to_show = append_metrics_row(df_to_show, metrics)
        return {
            "df": df_to_show,
            "text": df_to_show.to_string(index=True),
        }

    metrics_rows = []

    if "pred_quality" in df.columns and "odpad" in df.columns:
        regression_quality = {
            "model": "Quality",
            "metrics": calculate_regression_metrics(
                df[["pred_quality", "odpad"]].rename(columns={"pred_quality": "feature"}),
                "odpad",
            ),
        }
        metrics_rows.append(regression_quality)

    if "pred_delay" in df.columns and "lateness_h_sim" in df.columns:
        regression_delay = {
            "model": "Delay",
            "metrics": calculate_regression_metrics(
                df[["pred_delay", "lateness_h_sim"]].rename(columns={"pred_delay": "feature"}),
                "lateness_h_sim",
            ),
        }
        metrics_rows.append(regression_delay)

    if "recommended_machine" in df.columns and "machine_id" in df.columns:
        classification = {
            "model": "Schedule",
            "metrics": calculate_classification_metrics(
                df[["recommended_machine", "machine_id"]].rename(
                    columns={"recommended_machine": "feature"}
                ),
                "machine_id",
            ),
        }
        metrics_rows.append(classification)

    return metrics_rows


def load_training_data(path, train_ratio=0.8):
    df_full = load_csv(path)
    df_train, df_test = split_train_test(df_full, train_ratio=train_ratio)

    return {
        "full_df": df_full,
        "train_df": df_train,
        "test_df": df_test,
        "messages": [
            f"✔ Wczytano dane: {path}",
            f"Train: {len(df_train)} rekordów",
            f"Test: {len(df_test)} rekordów",
        ],
    }


def sanitize_filename(text: str) -> str:
    text = str(text).strip().replace(" ", "_")
    keep = []
    for char in text:
        if char.isalnum() or char in {"_", "-", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def build_model_filename(selected_models, metadata, backend="classic"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_part = "-".join(sorted([m.lower() for m in selected_models])) or "unknown"
    backend_part = sanitize_filename(backend.lower()) if backend else "classic"
    n_part = f"{metadata.get('n', 'x')}r"
    mach_part = f"{metadata.get('n_machines', 'x')}m"

    ksztalty = metadata.get("ksztalty", [])
    materialy = metadata.get("materialy", [])

    kszt_part = "-".join(ksztalty) if ksztalty else "allshapes"
    mat_part = "-".join(materialy) if materialy else "allmaterials"

    filename = (
        f"model_{backend_part}_{models_part}_{n_part}_{mach_part}_{kszt_part}_{mat_part}_{stamp}.pkl"
    )
    filename = sanitize_filename(filename)
    return MODELS_DIR / filename


def build_sto_model_filename(selected_methods):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods_part = "-".join(sorted([m.lower() for m in selected_methods])) or "unknown"
    filename = sanitize_filename(f"model_sto_{methods_part}_{stamp}.pkl")
    return MODELS_DIR / filename


def build_result_filename(model_name: str, source_name: str, suffix: str = ".csv") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_part = sanitize_filename(model_name.lower())
    source_part = sanitize_filename(source_name.lower())
    return DATA_DIR / f"wynik_priority_{model_part}_{source_part}_{stamp}{suffix}"


def train_models_flow(
    df_train,
    selected_models,
    metadata=None,
    progress_callback=None,
    backend="classic",
):
    if df_train is None or df_train.empty:
        raise ValueError("Brak danych treningowych.")
    if not selected_models:
        raise ValueError("Nie wybrano żadnego modelu do trenowania.")

    pack = train_selected_models(
        df_train=df_train,
        selected_models=selected_models,
        progress_callback=progress_callback,
        backend=backend,
    )
    pack["pack_kind"] = "ml"

    metadata = metadata or {}

    try:
        model_path = build_model_filename(selected_models, metadata, backend=backend)
    except TypeError:
        model_path = build_model_filename(selected_models, metadata)

    save_model_pack(pack, model_path)

    backend_label = "TabPFN" if backend == "tabpfn" else "Classic"

    return {
        "model_pack": pack,
        "model_path": model_path,
        "messages": [
            f"✔ Modele zapisane do: {model_path}",
            f"✔ Trening zakończony | backend: {backend_label}",
        ],
    }


def train_sto_models_flow(selected_methods):
    if not selected_methods:
        raise ValueError("Nie wybrano żadnego modelu STO do zapisania.")

    pack = {
        "pack_kind": "sto",
        "selected_methods": list(selected_methods),
    }

    model_path = build_sto_model_filename(selected_methods)
    save_model_pack(pack, model_path)

    return {
        "model_pack": pack,
        "model_path": model_path,
        "messages": [
            f"✔ Modele STO zapisane do: {model_path}",
            "✔ Zapis modelu STO zakończony",
        ],
    }


def solve_models_flow(model_path, data_path):
    if not model_path:
        raise ValueError("Nie wybrano pliku modelu.")
    if not data_path:
        raise ValueError("Nie wybrano pliku danych.")

    model_pack = load_model_pack(model_path)
    df = load_csv(data_path)

    if df.empty:
        raise ValueError("Plik danych jest pusty.")

    X, *_ = prepare_features(df)

    quality_model = model_pack.get("quality")
    delay_model = model_pack.get("delay")
    schedule_model = model_pack.get("schedule")
    scaler = model_pack.get("scaler")
    backend = model_pack.get("backend", "classic")

    X_for_pred = X

    if backend != "tabpfn" and scaler is not None:
        try:
            transformed = scaler.transform(X)
            if hasattr(X, "columns"):
                X_for_pred = pd.DataFrame(
                    transformed,
                    columns=X.columns,
                    index=X.index,
                )
            else:
                X_for_pred = transformed
        except Exception:
            X_for_pred = X

    result_df = df.copy()

    if quality_model is not None:
        result_df["pred_quality"] = quality_model.predict(X_for_pred)

    if delay_model is not None:
        result_df["pred_delay"] = delay_model.predict(X_for_pred)

    if quality_model is not None and delay_model is not None:
        result_df["priority"] = (
            result_df["pred_quality"] * 0.7
            + (1.0 / (1.0 + result_df["pred_delay"])) * 0.3
        )
        result_df = result_df.sort_values("priority", ascending=False).reset_index(drop=True)

    if schedule_model is not None:
        result_df["recommended_machine"] = simulate_schedule(schedule_model, result_df)

    model_name = Path(model_path).stem
    source_name = Path(data_path).stem
    result_path = build_result_filename(model_name, source_name, suffix=".csv")
    save_csv(result_df, result_path)

    return {
        "df": result_df,
        "result_path": result_path,
        "messages": [
            f"✔ Rozwiązanie gotowe: {result_path}",
        ],
    }


def analyze_sto_models(job_ids_text, processing_text, deadlines_text, selected_methods):
    jobs = parse_jobs(job_ids_text, processing_text, deadlines_text)
    results = run_selected_sto_models(jobs, selected_methods)
    report = build_sto_report(results)

    saved_paths = []
    best_result = None
    best_path = None

    if results:
        best_result = min(results, key=lambda x: x["sto"])
        best_jobs_df = apply_sto_result_to_dataframe(jobs, best_result)

        for result in results:
            result_df = apply_sto_result_to_dataframe(jobs, result)
            out_path = build_result_filename(
                f"sto_{result['method']}",
                "manual",
                suffix=".csv",
            )
            save_csv(result_df, out_path)
            saved_paths.append(
                {
                    "method": result["method"],
                    "sto": result["sto"],
                    "path": out_path,
                }
            )

        best_path = build_result_filename("sto_best", "manual", suffix=".csv")
        save_csv(best_jobs_df, best_path)

    return {
        "jobs": jobs,
        "results": results,
        "report": report,
        "saved_paths": saved_paths,
        "best_result": best_result,
        "best_path": best_path,
        "messages": ["✔ Analiza STO zakończona"],
    }


def solve_sto_with_saved_model(model_path, data_path):
    if not model_path:
        raise ValueError("Nie wybrano pliku modelu STO.")
    if not data_path:
        raise ValueError("Nie wybrano pliku danych STO.")

    model_pack = load_model_pack(model_path)
    if model_pack.get("pack_kind") != "sto":
        raise ValueError("Wybrany plik nie jest modelem STO.")

    df = load_csv(data_path)
    if df.empty:
        raise ValueError("Plik danych jest pusty.")

    jobs = dataframe_to_jobs(df)
    selected_methods = model_pack.get("selected_methods", [])
    if not selected_methods:
        raise ValueError("Zapisany model STO nie zawiera żadnych metod.")

    results = run_selected_sto_models(jobs, selected_methods)
    report = build_sto_report(results)

    saved_paths = []
    best_result = None
    best_path = None

    if results:
        best_result = min(results, key=lambda x: x["sto"])
        best_jobs_df = apply_sto_result_to_dataframe(jobs, best_result)

        for result in results:
            result_df = apply_sto_result_to_dataframe(jobs, result)
            out_path = build_result_filename(
                f"sto_{result['method']}",
                Path(data_path).stem,
                suffix=".csv",
            )
            save_csv(result_df, out_path)
            saved_paths.append(
                {
                    "method": result["method"],
                    "sto": result["sto"],
                    "path": out_path,
                }
            )

        best_path = build_result_filename("sto_best", Path(data_path).stem, suffix=".csv")
        save_csv(best_jobs_df, best_path)

    return {
        "jobs": jobs,
        "results": results,
        "report": report,
        "saved_paths": saved_paths,
        "best_result": best_result,
        "best_path": best_path,
        "messages": ["✔ Rozwiązanie STO zakończone"],
    }
