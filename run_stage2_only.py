#!/usr/bin/env python3
"""
Executa apenas o Stage 2 a partir de um stage1_results.csv já existente.
"""

from pathlib import Path
import argparse
import pandas as pd

from llm_pipeline.client import create_client, close_client
from llm_pipeline.data import (
    load_input,
    load_patterns_with_report,
    prepare_issue,
)
from llm_pipeline.stages import (
    run_stage2_sync,
    stage2_pairs_from_stage1,
)
from llm_pipeline.aggregation import aggregate_issue_results
from llm_pipeline.cli import _empty_stage2_dataframe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV original utilizado no pipeline",
    )

    parser.add_argument(
        "--patterns",
        required=True,
        help="CSV do catálogo de patterns",
    )

    parser.add_argument(
        "--stage1-results",
        required=True,
        help="Arquivo stage1_results.csv já gerado",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretório onde salvar os resultados",
    )

    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--stage2-model",
        default="gemini-2.5-pro",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--stage2-max-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--stage2-thinking-level",
        default="medium",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    patterns_path = Path(args.patterns)
    stage1_path = Path(args.stage1_results)
    run_dir = Path(args.output_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando CSV original...")
    df, _ = load_input(input_path)

    print("Carregando catálogo...")
    catalog, _ = load_patterns_with_report(patterns_path)

    print("Preparando artefatos...")
    prepared_issues = [
        prepare_issue(row, args.max_input_chars)
        for _, row in df.iterrows()
    ]

    prepared_by_key = {
        (x.repository, x.issue_number): x
        for x in prepared_issues
    }

    print("Carregando Stage 1...")
    stage1_df = pd.read_csv(stage1_path)

    print("Reconstruindo pares Stage 2...")
    pairs = stage2_pairs_from_stage1(stage1_df)

    print(f"{len(pairs)} pares encontrados.")

    client = create_client()

    try:

        if not pairs:
            stage2_df = _empty_stage2_dataframe()

        else:
            stage2_df = run_stage2_sync(
                client,
                prepared_by_key,
                pairs,
                catalog,
                args.stage2_model,
                args.temperature,
                args.stage2_max_tokens,
                args.seed,
                args.stage2_thinking_level,
                run_dir,
            )

        stage2_file = run_dir / "stage2_results.csv"

        stage2_df.to_csv(
            stage2_file,
            index=False,
        )

        print(f"Stage 2 salvo em: {stage2_file}")

        print("Gerando agregação final...")

        issue_df = aggregate_issue_results(
            stage1_df,
            stage2_df,
        )

        issue_file = run_dir / "issue_results.csv"

        issue_df.to_csv(
            issue_file,
            index=False,
        )

        print(f"Issue results salvo em: {issue_file}")

    finally:
        close_client(client)


if __name__ == "__main__":
    main()