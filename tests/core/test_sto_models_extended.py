import pytest

from AOA.core.sto_models import Job, parse_jobs, run_selected_sto_models, sequence_genetic


def test_parse_jobs_rejects_empty_inputs():
    with pytest.raises(ValueError, match="Wprowadź komplet danych"):
        parse_jobs("", "", "")


def test_parse_jobs_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Liczba zleceń, czasów i terminów musi być taka sama"):
        parse_jobs("A,B", "1", "2,3")


@pytest.mark.parametrize(
    ("processing_text", "deadlines_text"),
    [
        ("0,2", "4,5"),
        ("-1,2", "4,5"),
        ("1,2", "0,5"),
        ("1,2", "-4,5"),
    ],
)
def test_parse_jobs_rejects_non_positive_processing_or_deadline(processing_text, deadlines_text):
    with pytest.raises(ValueError, match="ujemne rzeczy|dodatnie"):
        parse_jobs("A,B", processing_text, deadlines_text)


def test_run_selected_sto_models_rejects_no_methods():
    jobs = parse_jobs("A,B,C", "2,1,3", "5,4,9")

    with pytest.raises(ValueError, match="Wybierz przynajmniej jeden model STO"):
        run_selected_sto_models(jobs, [])


def test_run_selected_sto_models_returns_result_for_each_method():
    jobs = parse_jobs("A,B,C", "2,1,3", "5,4,9")

    results = run_selected_sto_models(jobs, ["MT", "MO", "MZO", "GENETIC"])

    assert len(results) == 4
    for item in results:
        assert "method" in item
        assert "order" in item
        assert "steps" in item
        assert "sto" in item
        assert isinstance(item["order"], list)
        assert isinstance(item["steps"], list)


def test_sequence_genetic_for_small_input_returns_permutation_of_jobs():
    jobs = [
        Job("A", 2.0, 6.0),
        Job("B", 1.0, 3.0),
        Job("C", 4.0, 10.0),
    ]

    seq = sequence_genetic(jobs, seed=42)

    assert sorted(job.job_id for job in seq) == ["A", "B", "C"]


def test_sequence_genetic_for_large_input_returns_permutation_of_jobs():
    jobs = [Job(f"J{i}", float(i % 5 + 1), float(i + 5)) for i in range(1, 10)]

    seq = sequence_genetic(jobs, population_size=10, generations=5, seed=42)

    assert len(seq) == len(jobs)
    assert sorted(job.job_id for job in seq) == sorted(job.job_id for job in jobs)
