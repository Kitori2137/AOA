from AOA.core.sto_models import (
    build_sto_report,
    parse_jobs,
    run_selected_sto_models,
)


def test_parse_jobs_returns_three_jobs():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")

    assert len(jobs) == 3
    assert jobs[0].job_id == "Z1"
    assert jobs[1].processing_time == 20.0
    assert jobs[2].deadline == 110.0


def test_run_selected_sto_models_returns_sorted_results():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")
    results = run_selected_sto_models(jobs, ["MT", "MO", "MZO", "GENETIC"])

    assert len(results) == 4
    assert results[0]["sto"] <= results[-1]["sto"]
    assert all("method" in result for result in results)
    assert all("order" in result for result in results)
    assert all("steps" in result for result in results)


def test_build_sto_report_contains_best_and_worst_information():
    jobs = parse_jobs("Z1,Z2,Z3", "10,20,100", "150,30,110")
    results = run_selected_sto_models(jobs, ["MT", "MO", "MZO"])
    report = build_sto_report(jobs, results)

    assert "ANALIZA STO" in report
    assert "Najbardziej optymalny wynik:" in report
    assert "Najgorszy wynik:" in report
    assert "MODEL STO" in report
    assert "STO:" in report
