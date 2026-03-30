from datetime import datetime

from AOA.config import DATA_DIR, DEFAULT_RESULT_FILE, MODEL_FILE, MODELS_DIR
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
from AOA.core.sto_models import build_sto_report, parse_jobs, run_selected_sto_models

def build_main_page_summary(config: dict) -> str:
    selected_models = config.get("selected_models", [])
    selected_ksztalty = config.get("selected_ksztalty", [])
    selected_materialy = config.get("selected_materialy", [])

    return (
        "AKTUALNA KONFIGURACJA\n"
        "======================\n\n"
        f"Modele:\n  - {', '.join(selected_models) if selected_models else 'brak'}\n\n"
        f"Liczba rekordów: {config.get('n', '')}\n"
        f"Liczba maszyn: {config.get('n_machines', '')}\n"
        f"Test size: {config.get('test_size', '')}\n"
        f"Seed: {config.get('seed', '')}\n\n"
        f"Czas produkcji [h]: {config.get('prod_min', '')} -> {config.get('prod_max', '')}\n"
        f"Bufor terminu [h]: {config.get('deadline_min', '')} -> {config.get('deadline_max', '')}\n\n"
        f"Kształty:\n  - {', '.join(selected_ksztalty) if selected_ksztalty else 'brak'}\n\n"
        f"Materiały:\n  - {', '.join(selected_materialy) if selected_materialy else 'brak'}\n"
    )


def build_main_page_status(df_train=None, df_test=None) -> str:
    if df_train is None:
        return "Brak danych treningowych"

    return (
        "Dane treningowe gotowe\n"
        f"Train: {len(df_train)} rekordów\n"
        f"Test: {len(df_test) if df_test is not None else 0} rekordów"
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
            f"📄 Zapisano pełny zbiór: {full_path}",
            f"📄 Zapisano train: {train_path}",
            f"📄 Zapisano test: {test_path}",
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
        "messages": build_loaded_file_messages(path, df_train, df_test),
    }


def train_models_flow(df_train, selected_models, metadata=None, progress_callback=None):
    pack = train_selected_models(
        df_train=df_train,
        selected_models=selected_models,
        progress_callback=progress_callback,
    )

    model_path = build_model_filename(selected_models, metadata or {})
    save_model_pack(pack, model_path)

    return {
        "model_pack": pack,
        "model_path": model_path,
        "messages": [
            f"💾 Modele zapisane do: {model_path}",
            "✔ Trening zakończony",
        ],
    }


def solve_models_flow(model_path, data_path):
    pack = load_model_pack(model_path)
    df_sol = load_csv(data_path)

    X, _, _, _ = prepare_features(df_sol, pack.get("scaler"))

    if pack.get("quality") is not None:
        df_sol["pred_quality"] = pack["quality"].predict(X)

    if pack.get("delay") is not None:
        df_sol["pred_delay"] = pack["delay"].predict(X)

    if "pred_quality" in df_sol.columns and "pred_delay" in df_sol.columns:
        df_sol["priority"] = df_sol["pred_quality"] / (df_sol["pred_delay"] + 1e-6)
        df_sol = df_sol.sort_values("priority", ascending=False)

    df_sol = simulate_schedule(df_sol)
    save_csv(df_sol, DEFAULT_RESULT_FILE)

    return {
        "df": df_sol,
        "messages": [
            "🏁 Rozwiązanie gotowe",
            "TOP 10 produktów:",
            df_sol.head(10).to_string(),
            f"📄 Zapisano: {DEFAULT_RESULT_FILE}",
        ],
    }


def build_model_filename(selected_models, metadata):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    models_part = "-".join(sorted([m.lower() for m in selected_models])) or "unknown"
    n_part = f"{metadata.get('n', 'x')}r"
    mach_part = f"{metadata.get('n_machines', 'x')}m"

    ksztalty = metadata.get("ksztalty", [])
    materialy = metadata.get("materialy", [])

    kszt_part = "-".join(ksztalty) if ksztalty else "allshapes"
    mat_part = "-".join(materialy) if materialy else "allmaterials"

    filename = f"model_{models_part}_{n_part}_{mach_part}_{kszt_part}_{mat_part}_{stamp}.pkl"
    filename = sanitize_filename(filename)

    return MODELS_DIR / filename


def sanitize_filename(name: str) -> str:
    invalid = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ']
    for ch in invalid:
        name = name.replace(ch, "_")
    return name

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


def build_loaded_file_messages(path, df_train, df_test):
    return [
        f"✔ Wczytano dane: {path}",
        f"Train: {len(df_train)} rekordów",
        f"Test: {len(df_test)} rekordów",
    ]

def analyze_sto_models(job_ids_text, processing_text, deadlines_text, selected_methods):
    jobs = parse_jobs(job_ids_text, processing_text, deadlines_text)
    results = run_selected_sto_models(jobs, selected_methods)
    report = build_sto_report(jobs, results)

    return {
        "jobs": jobs,
        "results": results,
        "report": report,
    }
