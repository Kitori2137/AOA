import pytest

from AOA.core.sto_models import (
    build_sto_report,
    evaluate_sequence,
    parse_jobs,
    run_selected_sto_models,
    sequence_genetic,
    sequence_mo,
    sequence_mt,
    sequence_mzo,
)


def test_parse_jobs_returns_correct_number_of_jobs():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")

    assert len(jobs) == 3
    assert jobs[0].job_id == "Z1"
    assert jobs[1].processing_time == 20
    assert jobs[2].deadline == 110


def test_parse_jobs_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        parse_jobs("Z1,Z2", "10,20,30", "100,200")


def test_mt_mo_mzo_return_expected_sto_for_example():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")

    mt = evaluate_sequence(sequence_mt(jobs))
    mo = evaluate_sequence(sequence_mo(jobs))
    mzo = evaluate_sequence(sequence_mzo(jobs))

    assert mt["order"] == ["Z2", "Z3", "Z1"]
    assert mt["sto"] == 10

    assert mo["order"] == ["Z1", "Z2", "Z3"]
    assert mo["sto"] == 20

    assert mzo["order"] == ["Z3", "Z2", "Z1"]
    assert mzo["sto"] == 90


def test_sequence_genetic_for_small_input_matches_optimal_solution():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")
    genetic_sequence = sequence_genetic(jobs)

    result = evaluate_sequence(genetic_sequence)

    assert result["sto"] == 10
    assert result["order"] == ["Z2", "Z3", "Z1"]


def test_run_selected_sto_models_returns_sorted_results_best_first():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")
    results = run_selected_sto_models(jobs, ["MO", "MZO", "MT"])

    assert results[0]["method"] == "MT"
    assert results[0]["sto"] == 10

    assert results[1]["method"] == "MO"
    assert results[1]["sto"] == 20

    assert results[2]["method"] == "MZO"
    assert results[2]["sto"] == 90


def test_build_sto_report_contains_best_and_worst_information():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")
    results = run_selected_sto_models(jobs, ["MO", "MZO", "MT"])
    report = build_sto_report(jobs, results)

    assert "Najlepszy model: MT" in report
    assert "Najgorszy model: MZO" in report
    assert "STO = 10" in report
    assert "STO = 90" in report
