from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import random


@dataclass(frozen=True)
class Job:
    job_id: str
    processing_time: float
    deadline: float


def parse_jobs(job_ids_text: str, processing_text: str, deadlines_text: str) -> list[Job]:
    job_ids = [x.strip() for x in job_ids_text.split(",") if x.strip()]
    processing = [float(x.strip()) for x in processing_text.split(",") if x.strip()]
    deadlines = [float(x.strip()) for x in deadlines_text.split(",") if x.strip()]

    if not job_ids or not processing or not deadlines:
        raise ValueError("Wprowadź komplet danych: zlecenia, czasy i terminy.")

    if not (len(job_ids) == len(processing) == len(deadlines)):
        raise ValueError("Liczba zleceń, czasów i terminów musi być taka sama.")

    jobs = []
    for job_id, p, d in zip(job_ids, processing, deadlines):
        if p <= 0:
            raise ValueError(f"Czas wykonania dla {job_id} musi być > 0.")
        jobs.append(Job(job_id=job_id, processing_time=p, deadline=d))

    return jobs


def sequence_mt(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: (j.deadline, j.processing_time, j.job_id))


def sequence_mo(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: (j.processing_time, j.deadline, j.job_id))


def sequence_mzo(jobs: list[Job]) -> list[Job]:
    return sorted(jobs, key=lambda j: (-j.processing_time, j.deadline, j.job_id))


def evaluate_sequence(jobs: list[Job]) -> dict:
    t = 0.0
    steps = []
    sto = 0.0

    for job in jobs:
        t += job.processing_time
        lateness = max(0.0, t - job.deadline)
        sto += lateness
        steps.append({
            "job_id": job.job_id,
            "completion_time": t,
            "deadline": job.deadline,
            "lateness": lateness,
        })

    return {
        "order": [j.job_id for j in jobs],
        "steps": steps,
        "sto": sto,
    }


def sequence_genetic(
    jobs: list[Job],
    population_size: int = 40,
    generations: int = 80,
    mutation_rate: float = 0.12,
    seed: int = 42,
) -> list[Job]:
    if len(jobs) <= 8:
        # dla małych zestawów bierzemy optimum dokładne
        best_perm = min(permutations(jobs), key=lambda perm: evaluate_sequence(list(perm))["sto"])
        return list(best_perm)

    rng = random.Random(seed)

    def fitness(seq: list[Job]) -> float:
        return evaluate_sequence(seq)["sto"]

    def random_individual() -> list[Job]:
        seq = jobs[:]
        rng.shuffle(seq)
        return seq

    def crossover(parent1: list[Job], parent2: list[Job]) -> list[Job]:
        size = len(parent1)
        a, b = sorted(rng.sample(range(size), 2))
        middle = parent1[a:b]
        rest = [j for j in parent2 if j not in middle]
        return rest[:a] + middle + rest[a:]

    def mutate(seq: list[Job]) -> list[Job]:
        seq = seq[:]
        if rng.random() < mutation_rate and len(seq) >= 2:
            i, j = rng.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]
        return seq

    population = [random_individual() for _ in range(population_size)]

    for _ in range(generations):
        population = sorted(population, key=fitness)
        elites = population[: max(2, population_size // 5)]

        new_population = elites[:]
        while len(new_population) < population_size:
            p1 = rng.choice(elites)
            p2 = rng.choice(population[: max(4, population_size // 2)])
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    best = min(population, key=fitness)
    return best


def run_selected_sto_models(jobs: list[Job], selected_methods: list[str]) -> list[dict]:
    if not selected_methods:
        raise ValueError("Wybierz przynajmniej jeden model STO.")

    results = []

    mapping = {
        "MT": sequence_mt,
        "MO": sequence_mo,
        "MZO": sequence_mzo,
    }

    for method in selected_methods:
        if method == "GENETIC":
            seq = sequence_genetic(jobs)
        else:
            if method not in mapping:
                raise ValueError(f"Nieznany model: {method}")
            seq = mapping[method](jobs)

        result = evaluate_sequence(seq)
        result["method"] = method
        results.append(result)

    results.sort(key=lambda r: (r["sto"], r["method"]))
    return results


def build_sto_report(jobs: list[Job], results: list[dict]) -> str:
    lines = []
    lines.append("ANALIZA STO")
    lines.append("===========")
    lines.append("")
    lines.append("Dane wejściowe:")
    for job in jobs:
        lines.append(
            f"- {job.job_id}: czas={job.processing_time:g}, termin={job.deadline:g}"
        )

    lines.append("")
    lines.append("Wyniki modeli:")
    lines.append("")

    for idx, result in enumerate(results, start=1):
        lines.append(f"{idx}. {result['method']}")
        lines.append(f"   Kolejność: {', '.join(result['order'])}")
        lines.append(f"   STO: {result['sto']:g}")

        step_texts = []
        for step in result["steps"]:
            step_texts.append(
                f"{step['job_id']} -> C={step['completion_time']:g}, d={step['deadline']:g}, T+={step['lateness']:g}"
            )
        lines.append("   Kroki: " + " | ".join(step_texts))
        lines.append("")

    best = results[0]
    worst = results[-1]

    lines.append("Podsumowanie:")
    lines.append(f"- Najlepszy model: {best['method']} (STO = {best['sto']:g})")
    lines.append(f"- Najgorszy model: {worst['method']} (STO = {worst['sto']:g})")

    return "\n".join(lines)
