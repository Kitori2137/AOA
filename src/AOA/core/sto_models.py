from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random

import pandas as pd


@dataclass(frozen=True)
class Job:
    job_id: str
    processing_time: float
    deadline: float


def parse_jobs(job_ids_text: str, processing_text: str, deadlines_text: str) -> list[Job]:
    if not job_ids_text.strip() or not processing_text.strip() or not deadlines_text.strip():
        raise ValueError("Wprowadź komplet danych.")

    job_ids = [item.strip() for item in job_ids_text.split(",") if item.strip()]
    processing_values = [item.strip() for item in processing_text.split(",") if item.strip()]
    deadline_values = [item.strip() for item in deadlines_text.split(",") if item.strip()]

    if not (len(job_ids) == len(processing_values) == len(deadline_values)):
        raise ValueError("Liczba zleceń, czasów i terminów musi być taka sama.")

    jobs: list[Job] = []
    for job_id, p_raw, d_raw in zip(job_ids, processing_values, deadline_values):
        try:
            processing_time = float(p_raw)
            deadline = float(d_raw)
        except ValueError:
            raise ValueError("Czasy i terminy muszą być liczbami.") from None

        if processing_time <= 0 or deadline <= 0:
            raise ValueError("Czasy i terminy muszą być dodatnie.")

        jobs.append(Job(job_id=job_id, processing_time=processing_time, deadline=deadline))

    return jobs


def dataframe_to_jobs(
    df: pd.DataFrame,
    job_id_col: str | None = None,
    processing_col: str = "czas_produkcji_h",
    deadline_col: str = "termin_h",
    round_to_int: bool = True,
) -> list[Job]:
    if df is None or df.empty:
        raise ValueError("Brak danych do analizy STO.")

    if processing_col not in df.columns:
        raise ValueError(f"Brak kolumny czasu produkcji: {processing_col}")
    if deadline_col not in df.columns:
        raise ValueError(f"Brak kolumny terminu: {deadline_col}")

    jobs: list[Job] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        if job_id_col and job_id_col in df.columns:
            job_id = str(row[job_id_col])
        else:
            job_id = f"JOB_{idx + 1}"

        processing_time = float(row[processing_col])
        deadline = float(row[deadline_col])

        if round_to_int:
            processing_time = float(int(round(processing_time)))
            deadline = float(int(round(deadline)))

        if processing_time <= 0 or deadline <= 0:
            raise ValueError("Czasy i terminy w danych STO muszą być dodatnie.")

        jobs.append(
            Job(
                job_id=job_id,
                processing_time=processing_time,
                deadline=deadline,
            )
        )

    return jobs


def evaluate_sequence(sequence: list[Job]) -> dict:
    current_time = 0.0
    sto = 0.0
    steps: list[dict] = []

    for index, job in enumerate(sequence, start=1):
        start_time = current_time
        current_time += job.processing_time
        completion_time = current_time
        lateness = completion_time - job.deadline
        tardiness_positive = max(0.0, lateness)
        sto += tardiness_positive

        steps.append(
            {
                "lp": index,
                "job_id": job.job_id,
                "processing_time": round(job.processing_time, 3),
                "deadline": round(job.deadline, 3),
                "start_time": round(start_time, 3),
                "completion_time": round(completion_time, 3),
                "lateness": round(lateness, 3),
                "tardiness_positive": round(tardiness_positive, 3),
                "sto_running": round(sto, 3),
            }
        )

    return {
        "order": [job.job_id for job in sequence],
        "steps": steps,
        "sto": round(sto, 3),
        "completion_total": round(current_time, 3),
        "max_positive_delay": round(
            max((step["tardiness_positive"] for step in steps), default=0.0),
            3,
        ),
    }


def sequence_mt(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda job: (job.deadline, job.processing_time, job.job_id))


def sequence_mo(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda job: (job.processing_time, job.deadline, job.job_id))


def sequence_mzo(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda job: (-job.processing_time, job.deadline, job.job_id))


def _best_by_full_search(jobs: list[Job]) -> list[Job]:
    best_sequence = None
    best_eval = None

    for perm in permutations(jobs):
        seq = list(perm)
        evaluation = evaluate_sequence(seq)

        if best_eval is None:
            best_sequence = seq
            best_eval = evaluation
            continue

        if evaluation["sto"] < best_eval["sto"]:
            best_sequence = seq
            best_eval = evaluation
            continue

        if evaluation["sto"] == best_eval["sto"]:
            if evaluation["completion_total"] < best_eval["completion_total"]:
                best_sequence = seq
                best_eval = evaluation

    return best_sequence if best_sequence is not None else list(jobs)


def _mutate(order: list[Job], rng: random.Random) -> list[Job]:
    if len(order) < 2:
        return order[:]

    mutated = order[:]
    i, j = rng.sample(range(len(mutated)), 2)
    mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def sequence_genetic(
    jobs: list[Job],
    population_size: int = 30,
    generations: int = 40,
    seed: int = 42,
) -> list[Job]:
    if len(jobs) <= 8:
        return _best_by_full_search(jobs)

    rng = random.Random(seed)

    base_population: list[list[Job]] = [
        sequence_mt(jobs),
        sequence_mo(jobs),
        sequence_mzo(jobs),
        list(jobs),
    ]

    while len(base_population) < population_size:
        candidate = list(jobs)
        rng.shuffle(candidate)
        base_population.append(candidate)

    population = base_population[:population_size]

    for _ in range(generations):
        scored = sorted(
            ((evaluate_sequence(candidate)["sto"], candidate) for candidate in population),
            key=lambda item: item[0],
        )
        survivors = [candidate for _, candidate in scored[: max(4, population_size // 3)]]

        new_population = survivors[:]
        while len(new_population) < population_size:
            parent = rng.choice(survivors)
            child = _mutate(parent, rng)
            new_population.append(child)

        population = new_population

    best = min(population, key=lambda candidate: evaluate_sequence(candidate)["sto"])
    return best


def _format_steps_table(steps: list[dict]) -> str:
    if not steps:
        return "brak kroków"

    headers = ["Lp", "Zlecenie", "Czas", "Termin", "Start", "Koniec", "Opóźnienie", "T+", "STO nar."]
    rows = []

    for step in steps:
        rows.append(
            [
                str(step["lp"]),
                str(step["job_id"]),
                f'{step["processing_time"]:.3f}',
                f'{step["deadline"]:.3f}',
                f'{step["start_time"]:.3f}',
                f'{step["completion_time"]:.3f}',
                f'{step["lateness"]:.3f}',
                f'{step["tardiness_positive"]:.3f}',
                f'{step["sto_running"]:.3f}',
            ]
        )

    widths = []
    for col_index, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(row[col_index]))
        widths.append(max_width)

    def fmt_row(row_values: list[str]) -> str:
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row_values))

    separator = "-+-".join("-" * width for width in widths)

    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def run_selected_sto_models(jobs: list[Job], selected_methods: list[str]) -> list[dict]:
    if not selected_methods:
        raise ValueError("Wybierz przynajmniej jeden model STO.")

    method_map = {
        "MT": sequence_mt,
        "MO": sequence_mo,
        "MZO": sequence_mzo,
        "GENETIC": sequence_genetic,
    }

    results: list[dict] = []
    for method in selected_methods:
        if method not in method_map:
            raise ValueError(f"Nieznana metoda STO: {method}")

        sequence = method_map[method](jobs)
        evaluation = evaluate_sequence(sequence)

        result = {
            "method": method,
            "order": evaluation["order"],
            "steps": evaluation["steps"],
            "sto": evaluation["sto"],
            "completion_total": evaluation["completion_total"],
            "max_positive_delay": evaluation["max_positive_delay"],
        }
        result["table_text"] = _format_steps_table(result["steps"])
        results.append(result)

    results.sort(key=lambda item: (item["sto"], item["completion_total"], item["method"]))
    return results


def apply_sto_result_to_dataframe(
    df: pd.DataFrame,
    result: dict,
    job_id_col: str = "sto_job_id",
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Brak danych do zapisania wyniku STO.")

    out = df.copy().reset_index(drop=True)

    if job_id_col not in out.columns:
        out[job_id_col] = [f"JOB_{i + 1}" for i in range(len(out))]

    step_map = {step["job_id"]: step for step in result["steps"]}

    out = out.set_index(job_id_col)
    ordered_ids = [job_id for job_id in result["order"] if job_id in out.index]
    out = out.loc[ordered_ids].copy()

    out["sto_model"] = result["method"]
    out["sto_order"] = [step_map[job_id]["lp"] for job_id in ordered_ids]
    out["sto_start"] = [step_map[job_id]["start_time"] for job_id in ordered_ids]
    out["sto_end"] = [step_map[job_id]["completion_time"] for job_id in ordered_ids]
    out["sto_lateness"] = [step_map[job_id]["lateness"] for job_id in ordered_ids]
    out["sto_tardiness_positive"] = [
        step_map[job_id]["tardiness_positive"] for job_id in ordered_ids
    ]
    out["sto_cumulative"] = [step_map[job_id]["sto_running"] for job_id in ordered_ids]
    out["sto_total_for_model"] = result["sto"]

    return out.reset_index()


def build_sto_report(jobs: list[Job], results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("ANALIZA STO")
    lines.append("===========")
    lines.append("")
    lines.append("Dane wejściowe:")

    for job in jobs:
        lines.append(f"- {job.job_id}: czas={job.processing_time:g}, termin={job.deadline:g}")

    lines.append("")
    lines.append("Ranking modeli od najlepszego do najgorszego:")
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result['method']} | STO={result['sto']:.3f} | "
            f"Czas całk.={result['completion_total']:.3f} | "
            f"Max T+={result['max_positive_delay']:.3f}"
        )

    if results:
        best = results[0]
        worst = results[-1]

        lines.append("")
        lines.append("Najbardziej optymalny wynik:")
        lines.append(f"- Model: {best['method']}")
        lines.append(f"- Kolejność: {', '.join(best['order'])}")
        lines.append(f"- STO: {best['sto']:.3f}")

        lines.append("")
        lines.append("Najgorszy wynik:")
        lines.append(f"- Model: {worst['method']}")
        lines.append(f"- Kolejność: {', '.join(worst['order'])}")
        lines.append(f"- STO: {worst['sto']:.3f}")

    for index, result in enumerate(results, start=1):
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"{index}. MODEL STO: {result['method']}")
        lines.append("=" * 72)
        lines.append(f"Kolejność: {', '.join(result['order'])}")
        lines.append(f"STO: {result['sto']:.3f}")
        lines.append(f"Czas całkowity: {result['completion_total']:.3f}")
        lines.append(f"Maksymalne dodatnie opóźnienie: {result['max_positive_delay']:.3f}")
        lines.append("")
        lines.append("Szczegóły przypisania krok po kroku:")
        lines.append(result["table_text"])

    return "\n".join(lines)
