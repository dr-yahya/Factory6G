from __future__ import annotations

import json

from scripts.tools import generate_rm_ber_report as report_module


def test_rm_ber_report_ranks_by_ber_and_writes_outputs(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "resource_managers"
    stage_dir.mkdir(parents=True)
    stage_json = stage_dir / "stage_results_v2.json"
    summary_json = run_dir / "summary_v2.json"

    stage_json.write_text(
        json.dumps(
            {
                "run_id": "test_run",
                "stage": "resource_managers",
                "ebno_db_range": [0.0],
                "config_snapshot": {
                    "system": {
                        "channel_model_type": "tr38901",
                        "scenario": "umi",
                    }
                },
                "methods": {
                    "max_throughput": {"ber": [0.01], "ber_upper_confidence": [0.02]},
                    "ber_drl": {"ber": [0.005], "ber_upper_confidence": [0.01]},
                },
            }
        ),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(
            {
                "aggregate_means": {
                    "resource_managers": {
                        "max_throughput": {
                            "ber": 0.01,
                            "ber_upper_confidence": 0.02,
                            "throughput_bits_per_batch": 100.0,
                            "latency_ms": 2.0,
                            "avg_power_w": 0.1,
                            "num_batches": 1.0,
                            "bit_errors": 1.0,
                            "total_bits": 100.0,
                        },
                        "ber_drl": {
                            "ber": 0.005,
                            "ber_upper_confidence": 0.01,
                            "throughput_bits_per_batch": 120.0,
                            "latency_ms": 3.0,
                            "avg_power_w": 0.2,
                            "num_batches": 1.0,
                            "bit_errors": 1.0,
                            "total_bits": 200.0,
                        },
                    }
                },
                "runtime_totals_sec": {
                    "resource_managers": {
                        "max_throughput": 10.0,
                        "ber_drl": 11.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"
    rows = report_module.generate_report([stage_json], output_md, output_csv)

    ber_drl = next(row for row in rows if row.method == "ber_drl")
    max_throughput = next(row for row in rows if row.method == "max_throughput")
    assert ber_drl.rank_by_ber == 1
    assert max_throughput.rank_by_ber == 2
    assert output_csv.read_text(encoding="utf-8").splitlines()[0].startswith("run_id,channel_label")
    report_text = output_md.read_text(encoding="utf-8")
    assert "`ber_drl` beats best baseline `max_throughput`" in report_text
    assert "https://arxiv.org/abs/1705.09412" in report_text


def test_rm_ber_report_reranks_same_channel_across_stage_files(tmp_path):
    baseline_run = tmp_path / "baseline"
    trained_run = tmp_path / "trained"
    baseline_stage_dir = baseline_run / "resource_managers"
    trained_stage_dir = trained_run / "resource_managers"
    baseline_stage_dir.mkdir(parents=True)
    trained_stage_dir.mkdir(parents=True)
    baseline_stage_json = baseline_stage_dir / "stage_results_v2.json"
    trained_stage_json = trained_stage_dir / "stage_results_v2.json"

    common_config = {
        "system": {
            "channel_model_type": "tr38901",
            "scenario": "umi",
        }
    }
    baseline_stage_json.write_text(
        json.dumps(
            {
                "run_id": "baseline_run",
                "stage": "resource_managers",
                "ebno_db_range": [0.0],
                "config_snapshot": common_config,
                "methods": {
                    "max_throughput": {"ber": [0.01], "ber_upper_confidence": [0.02]},
                    "drl": {"ber": [0.02], "ber_upper_confidence": [0.03]},
                },
            }
        ),
        encoding="utf-8",
    )
    trained_stage_json.write_text(
        json.dumps(
            {
                "run_id": "trained_run",
                "stage": "resource_managers",
                "ebno_db_range": [0.0],
                "config_snapshot": common_config,
                "methods": {
                    "ber_drl": {"ber": [0.015], "ber_upper_confidence": [0.025]},
                },
            }
        ),
        encoding="utf-8",
    )

    rows = report_module.load_report_rows([baseline_stage_json, trained_stage_json])
    ranks = {row.method: row.rank_by_ber for row in rows}

    assert ranks == {
        "max_throughput": 1,
        "ber_drl": 2,
        "drl": 3,
    }
