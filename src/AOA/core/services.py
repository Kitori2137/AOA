from datetime import datetime
from pathlib import Path

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


def _parse_positive_int(value: str, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(POSITIVE_VALUES_MESSAGE)
    return parsed


def _parse_positive_float(value: str, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(POSITIVE_VALUES_MESSAGE)
    return parsed


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

    return (
        "AKTUALNA KONFIGURACJA\n"
        "======================\n\n"
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
    if df_train is None:
        return "Brak danych treningowych"

    return (
        "Dane treningowe gotowe\n"
        f"Train: {len(df_train)} rekordów\n"
        f"Test: {len(df_test) if df_test is not None else 0} rekordów"
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
    production_time_range=(1.0, 48.0),
    deadline_buffer_range=(1.0, 72.0),
):
    df_full, df_train, df_test = generate_production_data(
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
    full_path = DATA_DIR / f"production_data_{stamp}.csv"
    train_path = DATA_DIR / f"train_data_{stamp}.csv"
    test_path = DATA_DIR / f"test_data_{stamp}.csv"

    save_csv(df_full, full_path)
    save_csv(df_train, train_path)
    save_csv(df_test, test_path)

    return {
        "full_df": df_full,
        "train_df": df_train,
        "test_df": df_test,
        "full_path": full_path,
        "train_path": train_path,
        "test_path": test_path,
        "messages": [
            "✔ Dane wygenerowane i podzielone na train/test",
            f"Train: {len(df_train)} rekordów",
            f"Test: {len(df_test)} rekordów",
            f" Zapisano pełny zbiór: {full_path}",
            f" Zapisano train: {train_path}",
            f" Zapisano test: {test_path}",
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


def prepare_results_analysis(df, selected_cols, transformation, target, mode):
    if df is None or df.empty:
        raise ValueError("Brak danych do analizy")
    if not selected_cols:
        raise ValueError("Nie wybrano kolumn do analizy")

    df_to_show = df[selected_cols].copy()
    df_to_show = fill_missing_values(df_to_show)
    df_to_show = transform_numeric_columns(df_to_show, transformation)

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


def sanitize_filename(name: str) -> str:
    invalid = ['<', '>', ':', '"', "/", "\\", "|", "?", "*", " "]
    for ch in invalid:
        name = name.replace(ch, "_")
    return name


def build_model_filename(selected_models, metadata):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_part = "-".join(sorted([m.lower() for m in selected_models])) or "unknown"
    n_part = f"{metadata.get('n', 'x')}r"
    mach_part = f"{metadata.get('n_machines', 'x')}m"

    ksztalty = metadata.get("ksztalty", [])
    materialy = metadata.get("materialy", [])

    kszt_part = "-".join(ksztalty) if ksztalty else "allshapes"
    mat_part = "-".join(materialy) if materialy else "allmaterials"

    filename = (
        f"model_{models_part}_{n_part}_{mach_part}_{kszt_part}_{mat_part}_{stamp}.pkl"
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


def train_models_flow(df_train, selected_models, metadata=None, progress_callback=None):
    if df_train is None or df_train.empty:
        raise ValueError("Brak danych treningowych.")
    if not selected_models:
        raise ValueError("Nie wybrano żadnego modelu do trenowania.")

    pack = train_selected_models(
        df_train=df_train,
        selected_models=selected_models,
        progress_callback=progress_callback,
    )
    pack["pack_kind"] = "ml"

    model_path = build_model_filename(selected_models, metadata or {})
    save_model_pack(pack, model_path)

    return {
        "model_pack": pack,
        "model_path": model_path,
        "messages": [
            f" Modele zapisane do: {model_path}",
            "✔ Trening zakończony",
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
            f" Model STO zapisany do: {model_path}",
            f" Wybrane metody STO: {', '.join(selected_methods)}",
            "✔ Zapis konfiguracji STO zakończony",
        ],
    }


def solve_models_flow(model_path, data_path):
    if not model_path:
        raise ValueError("Nie wybrano pliku modelu.")
    if not data_path:
        raise ValueError("Nie wybrano pliku danych.")

    pack = load_model_pack(model_path)
    df_sol = load_csv(data_path)

    if df_sol is None or df_sol.empty:
        raise ValueError("Plik danych jest pusty.")

    pack_kind = pack.get("pack_kind", "ml")
    if pack_kind != "ml":
        raise ValueError("Wybrany plik nie jest paczką modeli ML.")

    X, _, _, _ = prepare_features(df_sol, pack.get("scaler"))

    if pack.get("quality") is not None:
        df_sol["pred_quality"] = pack["quality"].predict(X)

    if pack.get("delay") is not None:
        df_sol["pred_delay"] = pack["delay"].predict(X)

    if "pred_quality" in df_sol.columns and "pred_delay" in df_sol.columns:
        df_sol["priority"] = df_sol["pred_quality"] / (df_sol["pred_delay"] + 1e-6)
        df_sol = df_sol.sort_values("priority", ascending=False)

    selected_models = pack.get("selected_models", []) or []
    if "Schedule" in selected_models and pack.get("schedule") is not None:
        df_sol = simulate_schedule(df_sol)

    model_tag = "-".join([m.lower() for m in selected_models]) if selected_models else "ml"
    source_name = Path(str(data_path)).stem
    result_path = build_result_filename(model_tag, source_name)
    save_csv(df_sol, result_path)

    return {
        "df": df_sol,
        "result_path": result_path,
        "messages": [
            " Rozwiązanie gotowe",
            "TOP 10 produktów:",
            df_sol.head(10).to_string(),
            f" Zapisano: {result_path}",
        ],
    }


def analyze_sto_models(job_ids_text, processing_text, deadlines_text, selected_methods):
    jobs = parse_jobs(job_ids_text, processing_text, deadlines_text)
    results = run_selected_sto_models(jobs, selected_methods)
    report = build_sto_report(jobs, results)
    best_result = results[0] if results else None

    return {
        "jobs": jobs,
        "results": results,
        "report": report,
        "best_result": best_result,
    }


def solve_sto_with_saved_model(model_path, data_path):
    if not model_path:
        raise ValueError("Nie wybrano pliku modelu STO.")
    if not data_path:
        raise ValueError("Nie wybrano pliku danych.")

    pack = load_model_pack(model_path)
    pack_kind = pack.get("pack_kind")

    if pack_kind != "sto":
        raise ValueError("Wybrany plik nie jest zapisanym modelem STO.")

    selected_methods = pack.get("selected_methods", [])
    if not selected_methods:
        raise ValueError("Zapisany model STO nie zawiera metod.")

    df_source = load_csv(data_path)
    if df_source is None or df_source.empty:
        raise ValueError("Brak danych do rozwiązania STO.")

    df_source = df_source.copy().reset_index(drop=True)
    job_id_col = "sto_job_id"
    if job_id_col not in df_source.columns:
        df_source[job_id_col] = [f"JOB_{i + 1}" for i in range(len(df_source))]

    jobs = dataframe_to_jobs(
        df_source,
        job_id_col=job_id_col,
        processing_col="czas_produkcji_h",
        deadline_col="termin_h",
        round_to_int=True,
    )

    results = run_selected_sto_models(jobs, selected_methods)
    report = build_sto_report(jobs, results)

    source_name = Path(str(data_path)).stem
    saved_paths = []
    solved_frames = []

    for result in results:
        solved_df = apply_sto_result_to_dataframe(
            df=df_source,
            result=result,
            job_id_col=job_id_col,
        )
        result_path = build_result_filename(result["method"], source_name)
        save_csv(solved_df, result_path)

        saved_paths.append(
            {
                "method": result["method"],
                "path": result_path,
                "sto": result["sto"],
            }
        )
        solved_frames.append(
            {
                "method": result["method"],
                "df": solved_df,
                "sto": result["sto"],
            }
        )

    best_result = results[0] if results else None
    best_path = None

    if best_result is not None:
        best_df = apply_sto_result_to_dataframe(
            df=df_source,
            result=best_result,
            job_id_col=job_id_col,
        )
        best_path = build_result_filename(f"best_{best_result['method']}", source_name)
        save_csv(best_df, best_path)

    return {
        "jobs": jobs,
        "results": results,
        "report": report,
        "best_result": best_result,
        "saved_paths": saved_paths,
        "best_path": best_path,
        "solved_frames": solved_frames,
        "messages": [
            f"✔ Rozwiązano STO dla pliku: {data_path}",
            f"✔ Użyto zapisanych metod: {', '.join(selected_methods)}",
        ],
    }
