from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent
from typing import Iterable

from AOA.config import DATA_DIR, MODELS_DIR
from AOA.core.data_io import load_csv
from AOA.core.services import (
    analyze_sto_models,
    build_dataframe_preview_text,
    build_main_page_status,
    build_main_page_summary,
    generate_and_store_datasets,
    load_training_data,
    solve_models_flow,
    solve_sto_with_saved_model,
    split_selected_models,
    train_models_flow,
    train_sto_models_flow,
)
from AOA.utils.error_utils import write_exception_log


AVAILABLE_ML_MODELS = {"Quality", "Delay", "Schedule"}
AVAILABLE_STO_MODELS = {"MT", "MO", "MZO", "GENETIC"}
AVAILABLE_BACKENDS = {"classic", "tabpfn"}
DEFAULT_SHAPES = ["kwadrat", "trojkat", "trapez"]
DEFAULT_MATERIALS = ["bawelna", "mikrofibra", "poliester", "wiskoza"]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def validate_models(selected: Iterable[str], allowed: set[str], label: str) -> list[str]:
    selected = list(selected)
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        raise ValueError(
            f"Nieprawidłowe {label}: {', '.join(invalid)}. "
            f"Dozwolone: {', '.join(sorted(allowed))}"
        )
    return selected


def print_messages(messages: list[str] | None):
    if not messages:
        return
    for msg in messages:
        print(msg)


def print_key_value(title: str, value):
    print(f"{title}: {value}")


def hr(title: str):
    print(f"\n===== {title} =====")


def progress_callback(model_name: str, percent: float, detail: str = ""):
    suffix = f" | {detail}" if detail else ""
    print(f"[POSTĘP] {model_name}: {percent:.1f}%{suffix}")


def resolve_existing_file(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono {label}: {path}")
    return path


def resolve_optional_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(path_str)


def build_examples_text() -> str:
    return dedent(
        f"""
        Przykłady:

          aoa-cli generate --n 800 --machines 1 --test-size 0.2 --seed 42

          aoa-cli train --data data/train_x.csv --models Quality,Delay --backend classic

          aoa-cli train --data data/train_x.csv --models Quality --backend tabpfn

          aoa-cli solve --model models/model_x.pkl --data data/test_x.csv

          aoa-cli sto-run --jobs Z1,Z2,Z3 --times 10,20,100 --deadlines 150,30,110 --methods MT,MO,MZO

          aoa-cli sto-train --methods MT,MO,MZO,GENETIC

          aoa-cli sto-solve --model models/model_sto_x.pkl --data data/test_x.csv

          aoa-cli preview --data data/test_x.csv --rows 10

          aoa-cli summary --models Quality,Delay,MT --backend tabpfn --n 800 --machines 1

          aoa-cli status --data data/train_x.csv

          aoa-cli workflow --n 800 --models Quality,Delay --backend classic

          aoa-cli interactive
        """
    ).strip()


class ExamplesHelpFormatter(argparse.RawDescriptionHelpFormatter):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aoa-cli",
        description="CLI do AOA: generowanie danych, trening modeli ML/STO i rozwiązywanie bez GUI.",
        epilog=build_examples_text(),
        formatter_class=ExamplesHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser(
        "generate",
        help="Wygeneruj dane treningowe i testowe",
        description="Generuje pełny zbiór danych oraz automatycznie zapisuje full/train/test do katalogu data.",
        epilog="Przykład: aoa-cli generate --n 800 --machines 1 --test-size 0.2 --seed 42",
        formatter_class=ExamplesHelpFormatter,
    )
    gen_parser.add_argument("--n", type=int, default=5000, help="Liczba rekordów do wygenerowania")
    gen_parser.add_argument("--machines", type=int, default=1, help="Liczba maszyn")
    gen_parser.add_argument("--test-size", type=float, default=0.2, help="Udział zbioru testowego, np. 0.2")
    gen_parser.add_argument("--seed", type=int, default=42, help="Seed generatora")
    gen_parser.add_argument("--prod-min", type=float, default=1.0, help="Minimalny czas produkcji [h]")
    gen_parser.add_argument("--prod-max", type=float, default=48.0, help="Maksymalny czas produkcji [h]")
    gen_parser.add_argument("--deadline-min", type=float, default=1.0, help="Minimalny bufor terminu [h]")
    gen_parser.add_argument("--deadline-max", type=float, default=72.0, help="Maksymalny bufor terminu [h]")
    gen_parser.add_argument(
        "--shapes",
        type=str,
        default=",".join(DEFAULT_SHAPES),
        help="Lista kształtów po przecinku",
    )
    gen_parser.add_argument(
        "--materials",
        type=str,
        default=",".join(DEFAULT_MATERIALS),
        help="Lista materiałów po przecinku",
    )

    # train
    train_parser = subparsers.add_parser(
        "train",
        help="Wytrenuj i zapisz modele ML lub STO",
        description="Wczytuje CSV, dzieli dane na train/test i zapisuje wytrenowane modele.",
        epilog="Przykład: aoa-cli train --data data/train_x.csv --models Quality,Delay --backend classic",
        formatter_class=ExamplesHelpFormatter,
    )
    train_parser.add_argument("--data", required=True, type=str, help="Ścieżka do CSV z danymi")
    train_parser.add_argument(
        "--models",
        required=True,
        type=str,
        help="Lista modeli po przecinku, np. Quality,Delay albo MT,MO",
    )
    train_parser.add_argument(
        "--backend",
        choices=sorted(AVAILABLE_BACKENDS),
        default="classic",
        help="Backend ML: classic albo tabpfn",
    )
    train_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Podział train/test przy wczytaniu CSV",
    )
    train_parser.add_argument("--n-meta", type=int, default=None, help="Opcjonalne n do nazwy modelu")
    train_parser.add_argument("--machines-meta", type=int, default=None, help="Opcjonalne n_machines do nazwy modelu")
    train_parser.add_argument("--shapes-meta", type=str, default="", help="Opcjonalne kształty do nazwy modelu")
    train_parser.add_argument("--materials-meta", type=str, default="", help="Opcjonalne materiały do nazwy modelu")

    # solve
    solve_parser = subparsers.add_parser(
        "solve",
        help="Rozwiąż zapisanym modelem ML",
        description="Wczytuje zapisany model ML i tworzy wynikowy CSV dla wskazanych danych.",
        epilog="Przykład: aoa-cli solve --model models/model_x.pkl --data data/test_x.csv",
        formatter_class=ExamplesHelpFormatter,
    )
    solve_parser.add_argument("--model", required=True, type=str, help="Ścieżka do pliku modelu .pkl")
    solve_parser.add_argument("--data", required=True, type=str, help="Ścieżka do pliku CSV")

    # sto-run
    sto_run_parser = subparsers.add_parser(
        "sto-run",
        help="Uruchom ręczną analizę STO",
        description="Liczy metody STO na danych wpisanych bezpośrednio z terminala.",
        epilog="Przykład: aoa-cli sto-run --jobs Z1,Z2,Z3 --times 10,20,100 --deadlines 150,30,110 --methods MT,MO,MZO",
        formatter_class=ExamplesHelpFormatter,
    )
    sto_run_parser.add_argument("--jobs", required=True, type=str, help="Lista zleceń, np. Z1,Z2,Z3")
    sto_run_parser.add_argument("--times", required=True, type=str, help="Lista czasów, np. 10,20,100")
    sto_run_parser.add_argument("--deadlines", required=True, type=str, help="Lista terminów, np. 150,30,110")
    sto_run_parser.add_argument("--methods", required=True, type=str, help="Lista metod STO, np. MT,MO,MZO")

    # sto-train
    sto_train_parser = subparsers.add_parser(
        "sto-train",
        help="Zapisz paczkę modeli STO",
        description="Zapisuje paczkę metod STO do późniejszego użycia.",
        epilog="Przykład: aoa-cli sto-train --methods MT,MO,MZO,GENETIC",
        formatter_class=ExamplesHelpFormatter,
    )
    sto_train_parser.add_argument("--methods", required=True, type=str, help="Lista metod STO")

    # sto-solve
    sto_solve_parser = subparsers.add_parser(
        "sto-solve",
        help="Rozwiąż zapisanym modelem STO",
        description="Wczytuje paczkę STO i tworzy wynikowe CSV dla wskazanych danych.",
        epilog="Przykład: aoa-cli sto-solve --model models/model_sto_x.pkl --data data/test_x.csv",
        formatter_class=ExamplesHelpFormatter,
    )
    sto_solve_parser.add_argument("--model", required=True, type=str, help="Ścieżka do pliku modelu STO .pkl")
    sto_solve_parser.add_argument("--data", required=True, type=str, help="Ścieżka do pliku CSV")

    # preview
    preview_parser = subparsers.add_parser(
        "preview",
        help="Pokaż podgląd CSV",
        description="Wyświetla podgląd danych z pliku CSV w terminalu.",
        epilog="Przykład: aoa-cli preview --data data/test_x.csv --rows 10",
        formatter_class=ExamplesHelpFormatter,
    )
    preview_parser.add_argument("--data", required=True, type=str, help="Ścieżka do pliku CSV")
    preview_parser.add_argument("--rows", type=int, default=15, help="Maksymalna liczba wierszy podglądu")

    # summary
    summary_parser = subparsers.add_parser(
        "summary",
        help="Pokaż podsumowanie konfiguracji jak w GUI",
        description="Buduje tekstowe podsumowanie podobne do panelu bocznego w GUI.",
        epilog="Przykład: aoa-cli summary --models Quality,Delay,MT --backend tabpfn --n 800 --machines 1",
        formatter_class=ExamplesHelpFormatter,
    )
    summary_parser.add_argument("--models", required=True, type=str, help="Lista modeli po przecinku")
    summary_parser.add_argument("--backend", choices=sorted(AVAILABLE_BACKENDS), default="classic", help="Backend ML")
    summary_parser.add_argument("--n", type=int, default=5000, help="Liczba rekordów")
    summary_parser.add_argument("--machines", type=int, default=1, help="Liczba maszyn")
    summary_parser.add_argument("--test-size", type=float, default=0.2, help="Udział zbioru testowego")
    summary_parser.add_argument("--seed", type=int, default=42, help="Seed")
    summary_parser.add_argument("--prod-min", type=float, default=1.0, help="Minimalny czas produkcji [h]")
    summary_parser.add_argument("--prod-max", type=float, default=48.0, help="Maksymalny czas produkcji [h]")
    summary_parser.add_argument("--deadline-min", type=float, default=1.0, help="Minimalny bufor terminu [h]")
    summary_parser.add_argument("--deadline-max", type=float, default=72.0, help="Maksymalny bufor terminu [h]")
    summary_parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES), help="Lista kształtów")
    summary_parser.add_argument("--materials", type=str, default=",".join(DEFAULT_MATERIALS), help="Lista materiałów")

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Pokaż status danych",
        description="Pokazuje status danych podobny do GUI. Gdy podasz CSV, liczy train/test z podziałem.",
        epilog="Przykład: aoa-cli status --data data/train_x.csv",
        formatter_class=ExamplesHelpFormatter,
    )
    status_parser.add_argument("--data", type=str, default="", help="Opcjonalna ścieżka do pliku CSV")
    status_parser.add_argument("--train-ratio", type=float, default=0.8, help="Podział train/test")

    # workflow
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Wykonaj cały przepływ: generate -> train -> solve",
        description="Komenda zbiorcza dla pracy bez GUI: generuje dane, trenuje modele i rozwiązuje test.",
        epilog="Przykład: aoa-cli workflow --n 800 --models Quality,Delay --backend classic",
        formatter_class=ExamplesHelpFormatter,
    )
    workflow_parser.add_argument("--n", type=int, default=800, help="Liczba rekordów")
    workflow_parser.add_argument("--machines", type=int, default=1, help="Liczba maszyn")
    workflow_parser.add_argument("--test-size", type=float, default=0.2, help="Udział zbioru testowego")
    workflow_parser.add_argument("--seed", type=int, default=42, help="Seed")
    workflow_parser.add_argument("--prod-min", type=float, default=1.0, help="Minimalny czas produkcji [h]")
    workflow_parser.add_argument("--prod-max", type=float, default=48.0, help="Maksymalny czas produkcji [h]")
    workflow_parser.add_argument("--deadline-min", type=float, default=1.0, help="Minimalny bufor terminu [h]")
    workflow_parser.add_argument("--deadline-max", type=float, default=72.0, help="Maksymalny bufor terminu [h]")
    workflow_parser.add_argument("--shapes", type=str, default=",".join(DEFAULT_SHAPES), help="Lista kształtów")
    workflow_parser.add_argument("--materials", type=str, default=",".join(DEFAULT_MATERIALS), help="Lista materiałów")
    workflow_parser.add_argument("--models", required=True, type=str, help="Lista modeli, np. Quality,Delay")
    workflow_parser.add_argument(
        "--backend",
        choices=sorted(AVAILABLE_BACKENDS),
        default="classic",
        help="Backend ML",
    )
    workflow_parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Tylko wygeneruj i wytrenuj, bez rozwiązywania testu",
    )

    # interactive
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Uruchom prosty tryb interaktywny",
        description="Proste menu tekstowe do pracy z terminala bez wpisywania pełnych komend.",
        formatter_class=ExamplesHelpFormatter,
    )
    interactive_parser.add_argument(
        "--quick",
        action="store_true",
        help="Tryb skrócony z domyślnymi wartościami tam, gdzie to możliwe",
    )

    return parser


def command_generate(args: argparse.Namespace) -> int:
    shapes = parse_csv_list(args.shapes)
    materials = parse_csv_list(args.materials)

    result = generate_and_store_datasets(
        n=args.n,
        n_machines=args.machines,
        test_size=args.test_size,
        seed=args.seed,
        ksztalty=shapes,
        materialy=materials,
        production_time_range=(args.prod_min, args.prod_max),
        deadline_buffer_range=(args.deadline_min, args.deadline_max),
    )

    hr("GENEROWANIE")
    print_messages(result.get("messages"))
    print_key_value("FULL", result["full_path"])
    print_key_value("TRAIN", result["train_path"])
    print_key_value("TEST", result["test_path"])
    return 0


def command_train(args: argparse.Namespace) -> int:
    data_path = resolve_existing_file(args.data, "pliku danych")
    selected_models = parse_csv_list(args.models)

    if not selected_models:
        raise ValueError("Nie podano żadnych modeli.")

    ml_models = [m for m in selected_models if m in AVAILABLE_ML_MODELS]
    sto_models = [m for m in selected_models if m in AVAILABLE_STO_MODELS]
    invalid_models = [m for m in selected_models if m not in AVAILABLE_ML_MODELS | AVAILABLE_STO_MODELS]

    if invalid_models:
        raise ValueError(
            f"Nieprawidłowe modele: {', '.join(invalid_models)}. "
            f"ML: {', '.join(sorted(AVAILABLE_ML_MODELS))}. "
            f"STO: {', '.join(sorted(AVAILABLE_STO_MODELS))}."
        )

    loaded = load_training_data(path=str(data_path), train_ratio=args.train_ratio)
    df_train = loaded["train_df"]

    hr("WCZYTANIE DANYCH")
    print_messages(loaded.get("messages"))

    if ml_models:
        backend = args.backend
        metadata = {
            "n": args.n_meta if args.n_meta is not None else len(loaded["full_df"]),
            "n_machines": args.machines_meta if args.machines_meta is not None else "x",
            "ksztalty": parse_csv_list(args.shapes_meta),
            "materialy": parse_csv_list(args.materials_meta),
        }

        hr("TRENING ML")
        print(f"Modele ML: {', '.join(ml_models)}")
        print(f"Backend: {backend}")

        result = train_models_flow(
            df_train=df_train,
            selected_models=ml_models,
            metadata=metadata,
            progress_callback=progress_callback,
            backend=backend,
        )
        print_messages(result.get("messages"))
        print_key_value("MODEL_ML", result["model_path"])

    if sto_models:
        hr("ZAPIS STO")
        print(f"Metody STO: {', '.join(sto_models)}")
        result = train_sto_models_flow(sto_models)
        print_messages(result.get("messages"))
        print_key_value("MODEL_STO", result["model_path"])

    return 0


def command_solve(args: argparse.Namespace) -> int:
    model_path = resolve_existing_file(args.model, "pliku modelu")
    data_path = resolve_existing_file(args.data, "pliku danych")

    hr("ROZWIĄZYWANIE ML")
    print_key_value("MODEL", model_path)
    print_key_value("DATA", data_path)

    result = solve_models_flow(model_path=str(model_path), data_path=str(data_path))
    print_messages(result.get("messages"))
    print_key_value("WYNIK", result["result_path"])
    return 0


def command_sto_run(args: argparse.Namespace) -> int:
    methods = validate_models(parse_csv_list(args.methods), AVAILABLE_STO_MODELS, "metody STO")

    result = analyze_sto_models(
        job_ids_text=args.jobs,
        processing_text=args.times,
        deadlines_text=args.deadlines,
        selected_methods=methods,
    )

    print_messages(result.get("messages"))
    hr("RAPORT STO")
    print(result["report"])

    if result.get("saved_paths"):
        hr("ZAPISANE PLIKI")
        for item in result["saved_paths"]:
            print(f"{item['method']}: {item['path']} | STO={item['sto']:.3f}")

    if result.get("best_path"):
        print(f"\nBEST: {result['best_path']}")

    return 0


def command_sto_train(args: argparse.Namespace) -> int:
    methods = validate_models(parse_csv_list(args.methods), AVAILABLE_STO_MODELS, "metody STO")
    result = train_sto_models_flow(methods)
    hr("ZAPIS STO")
    print_messages(result.get("messages"))
    print_key_value("MODEL_STO", result["model_path"])
    return 0


def command_sto_solve(args: argparse.Namespace) -> int:
    model_path = resolve_existing_file(args.model, "pliku modelu STO")
    data_path = resolve_existing_file(args.data, "pliku danych STO")

    result = solve_sto_with_saved_model(model_path=str(model_path), data_path=str(data_path))
    print_messages(result.get("messages"))
    hr("RAPORT STO")
    print(result["report"])

    if result.get("saved_paths"):
        hr("ZAPISANE PLIKI")
        for item in result["saved_paths"]:
            print(f"{item['method']}: {item['path']} | STO={item['sto']:.3f}")

    if result.get("best_path"):
        print(f"\nBEST: {result['best_path']}")

    return 0


def command_preview(args: argparse.Namespace) -> int:
    data_path = resolve_existing_file(args.data, "pliku danych")
    df = load_csv(data_path)
    print(build_dataframe_preview_text(df, title=f"Podgląd: {data_path.name}", max_rows=args.rows))
    return 0


def command_summary(args: argparse.Namespace) -> int:
    selected_models = parse_csv_list(args.models)
    ml_models, sto_models = split_selected_models(selected_models)
    invalid = [m for m in selected_models if m not in AVAILABLE_ML_MODELS | AVAILABLE_STO_MODELS]
    if invalid:
        raise ValueError(f"Nieprawidłowe modele: {', '.join(invalid)}")

    config = {
        "selected_models": ml_models + sto_models,
        "selected_ksztalty": parse_csv_list(args.shapes),
        "selected_materialy": parse_csv_list(args.materials),
        "n": args.n,
        "n_machines": args.machines,
        "test_size": args.test_size,
        "seed": args.seed,
        "prod_min": args.prod_min,
        "prod_max": args.prod_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "backend": args.backend,
    }

    hr("PODSUMOWANIE KONFIGURACJI")
    print(build_main_page_summary(config))
    return 0


def command_status(args: argparse.Namespace) -> int:
    if not args.data:
        hr("STATUS")
        print(build_main_page_status(None, None))
        return 0

    data_path = resolve_existing_file(args.data, "pliku danych")
    loaded = load_training_data(path=str(data_path), train_ratio=args.train_ratio)

    hr("STATUS")
    print(build_main_page_status(loaded["train_df"], loaded["test_df"]))
    print_key_value("ŹRÓDŁO", data_path)
    return 0


def command_workflow(args: argparse.Namespace) -> int:
    selected_models = parse_csv_list(args.models)
    if not selected_models:
        raise ValueError("Nie podano modeli do workflow.")

    ml_models = [m for m in selected_models if m in AVAILABLE_ML_MODELS]
    sto_models = [m for m in selected_models if m in AVAILABLE_STO_MODELS]
    invalid_models = [m for m in selected_models if m not in AVAILABLE_ML_MODELS | AVAILABLE_STO_MODELS]
    if invalid_models:
        raise ValueError(f"Nieprawidłowe modele: {', '.join(invalid_models)}")

    hr("WORKFLOW | GENEROWANIE")
    gen_result = generate_and_store_datasets(
        n=args.n,
        n_machines=args.machines,
        test_size=args.test_size,
        seed=args.seed,
        ksztalty=parse_csv_list(args.shapes),
        materialy=parse_csv_list(args.materials),
        production_time_range=(args.prod_min, args.prod_max),
        deadline_buffer_range=(args.deadline_min, args.deadline_max),
    )
    print_messages(gen_result.get("messages"))

    ml_model_path = None

    if ml_models:
        hr("WORKFLOW | TRENING ML")
        train_result = train_models_flow(
            df_train=gen_result["train_df"],
            selected_models=ml_models,
            metadata={
                "n": args.n,
                "n_machines": args.machines,
                "ksztalty": parse_csv_list(args.shapes),
                "materialy": parse_csv_list(args.materials),
            },
            progress_callback=progress_callback,
            backend=args.backend,
        )
        print_messages(train_result.get("messages"))
        ml_model_path = train_result["model_path"]

    if sto_models:
        hr("WORKFLOW | ZAPIS STO")
        sto_result = train_sto_models_flow(sto_models)
        print_messages(sto_result.get("messages"))

    if ml_models and not args.skip_solve and ml_model_path is not None:
        hr("WORKFLOW | ROZWIĄZYWANIE TESTU")
        solve_result = solve_models_flow(
            model_path=str(ml_model_path),
            data_path=str(gen_result["test_path"]),
        )
        print_messages(solve_result.get("messages"))
        print_key_value("WYNIK_TEST", solve_result["result_path"])

    hr("WORKFLOW | KONIEC")
    print_key_value("FULL", gen_result["full_path"])
    print_key_value("TRAIN", gen_result["train_path"])
    print_key_value("TEST", gen_result["test_path"])
    if ml_model_path is not None:
        print_key_value("MODEL_ML", ml_model_path)
    return 0


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def interactive_generate():
    args = argparse.Namespace(
        n=int(ask("Liczba rekordów", "800")),
        machines=int(ask("Liczba maszyn", "1")),
        test_size=float(ask("Test size", "0.2")),
        seed=int(ask("Seed", "42")),
        prod_min=float(ask("Czas prod. min", "1")),
        prod_max=float(ask("Czas prod. max", "48")),
        deadline_min=float(ask("Bufor terminu min", "1")),
        deadline_max=float(ask("Bufor terminu max", "72")),
        shapes=ask("Kształty", ",".join(DEFAULT_SHAPES)),
        materials=ask("Materiały", ",".join(DEFAULT_MATERIALS)),
    )
    return command_generate(args)


def interactive_train():
    args = argparse.Namespace(
        data=ask("Ścieżka do CSV z danymi"),
        models=ask("Modele", "Quality"),
        backend=ask("Backend ML", "classic"),
        train_ratio=float(ask("Train ratio", "0.8")),
        n_meta="",
        machines_meta="",
        shapes_meta="",
        materials_meta="",
    )

    args.n_meta = None
    args.machines_meta = None
    return command_train(args)


def interactive_solve():
    args = argparse.Namespace(
        model=ask("Ścieżka do modelu .pkl"),
        data=ask("Ścieżka do danych CSV"),
    )
    return command_solve(args)


def interactive_sto_run():
    args = argparse.Namespace(
        jobs=ask("Zlecenia", "Z1,Z2,Z3"),
        times=ask("Czasy", "10,20,100"),
        deadlines=ask("Terminy", "150,30,110"),
        methods=ask("Metody", "MT,MO,MZO"),
    )
    return command_sto_run(args)


def interactive_summary():
    args = argparse.Namespace(
        models=ask("Modele", "Quality,Delay"),
        backend=ask("Backend", "classic"),
        n=int(ask("Liczba rekordów", "800")),
        machines=int(ask("Liczba maszyn", "1")),
        test_size=float(ask("Test size", "0.2")),
        seed=int(ask("Seed", "42")),
        prod_min=float(ask("Czas prod. min", "1")),
        prod_max=float(ask("Czas prod. max", "48")),
        deadline_min=float(ask("Bufor terminu min", "1")),
        deadline_max=float(ask("Bufor terminu max", "72")),
        shapes=ask("Kształty", ",".join(DEFAULT_SHAPES)),
        materials=ask("Materiały", ",".join(DEFAULT_MATERIALS)),
    )
    return command_summary(args)


def interactive_status():
    args = argparse.Namespace(
        data=ask("Ścieżka do CSV (puste = brak danych)", ""),
        train_ratio=float(ask("Train ratio", "0.8")),
    )
    return command_status(args)


def interactive_workflow():
    args = argparse.Namespace(
        n=int(ask("Liczba rekordów", "800")),
        machines=int(ask("Liczba maszyn", "1")),
        test_size=float(ask("Test size", "0.2")),
        seed=int(ask("Seed", "42")),
        prod_min=float(ask("Czas prod. min", "1")),
        prod_max=float(ask("Czas prod. max", "48")),
        deadline_min=float(ask("Bufor terminu min", "1")),
        deadline_max=float(ask("Bufor terminu max", "72")),
        shapes=ask("Kształty", ",".join(DEFAULT_SHAPES)),
        materials=ask("Materiały", ",".join(DEFAULT_MATERIALS)),
        models=ask("Modele", "Quality"),
        backend=ask("Backend", "classic"),
        skip_solve=ask("Pominąć solve? tak/nie", "nie").lower() in {"tak", "t", "yes", "y"},
    )
    return command_workflow(args)


def command_interactive(args: argparse.Namespace) -> int:
    while True:
        hr("TRYB INTERAKTYWNY")
        print("1. Generate")
        print("2. Train")
        print("3. Solve")
        print("4. STO run")
        print("5. Summary")
        print("6. Status")
        print("7. Workflow")
        print("0. Wyjście")

        choice = input("Wybierz opcję: ").strip()

        if choice == "1":
            interactive_generate()
        elif choice == "2":
            interactive_train()
        elif choice == "3":
            interactive_solve()
        elif choice == "4":
            interactive_sto_run()
        elif choice == "5":
            interactive_summary()
        elif choice == "6":
            interactive_status()
        elif choice == "7":
            interactive_workflow()
        elif choice == "0":
            print("Koniec.")
            return 0
        else:
            print("Nieprawidłowy wybór.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "generate":
            return command_generate(args)
        if args.command == "train":
            return command_train(args)
        if args.command == "solve":
            return command_solve(args)
        if args.command == "sto-run":
            return command_sto_run(args)
        if args.command == "sto-train":
            return command_sto_train(args)
        if args.command == "sto-solve":
            return command_sto_solve(args)
        if args.command == "preview":
            return command_preview(args)
        if args.command == "summary":
            return command_summary(args)
        if args.command == "status":
            return command_status(args)
        if args.command == "workflow":
            return command_workflow(args)
        if args.command == "interactive":
            return command_interactive(args)

        parser.print_help()
        return 1

    except KeyboardInterrupt:
        eprint("Przerwano przez użytkownika.")
        return 130
    except Exception as e:
        log_path = write_exception_log("cli.main", e)
        eprint(f"❌ Błąd: {type(e).__name__}: {e}")
        eprint(f"Szczegóły zapisano w: {log_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
