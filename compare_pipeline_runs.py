#!/usr/bin/env python3
"""Compare persisted A/B runs at retrieval, pair, and final issue levels.

The command is intentionally offline. It consumes existing artifacts, never
creates Gemini requests, and refuses to replace an existing comparison.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from evaluate_pipeline import binary_metrics
from llm_pipeline.checkpoint import ResultStore
from llm_pipeline.data import load_input, load_patterns, prepare_issue
from llm_pipeline.human_annotations import load_human_gold
from llm_pipeline.issue_evaluation import ISSUE_KEY, human_issue_reference, issue_level_binary
from llm_pipeline.normalization import read_csv_robust
from llm_pipeline.stages import pipeline_integrity_report, stage2_pairs_from_stage1
from llm_pipeline.utils import sha256_file, write_json

PAIR_KEY = ["repository", "issue_number", "pattern"]
NOT_MEASURED = "NOT_MEASURED"

# Evidence-based audit labels for this frozen pilot. These labels annotate only
# the report; they never change predictions, human labels, or pipeline behavior.
PILOT_DIAGNOSES: dict[str, tuple[str, str]] = {
    "ProjectOpenSea/opensea-js#236": ("false_friend", "Wyvern user proxy is not the catalog's upgrade/delegation proxy mechanism."),
    "Consensys/gnark#1311": ("background_architecture", "ZK proof is library context; the PR focuses on hashing public inputs for calldata/recursion."),
    "MoralisWeb3/react-moralis#142": ("human_model_disagreement", "ENS resolution resembles a registry, but the reviewed annotation excludes it from the catalog pattern."),
    "matter-labs/foundry-zksync#358": ("false_friend", "Test-runner account migration/initialization is not blockchain state initialization as defined by the catalog."),
    "OpenZeppelin/openzeppelin-contracts#4084": ("background_architecture", "AccessControl is only listed as a target of formal-verification tooling."),
    "Uniswap/v3-sdk#141": ("insufficient_pattern_discussion", "CREATE2 address derivation is discussed without a creator/factory contract mechanism."),
    "paritytech/polkadot-sdk#4745": ("false_friend", "Runtime storage migration is not source-to-destination blockchain state initialization."),
    "MetaMask/metamask-mobile#4441": ("insufficient_pattern_discussion", "The artifact is a wallet typed-data signing UI bug, not the selected catalog mechanism."),
    "paritytech/substrate#9738": ("background_architecture", "Off-chain consumers are context; the PR changes event-record encoding/decoding."),
    "trustwallet/assets#4740": ("human_model_disagreement", "A bot's payment-to-PR automation matches Reverse Oracle textually but is peripheral to the annotated PR topic."),
    "near/core-contracts#197": ("background_architecture", "The token standard is test context rather than a discussion of tokenization."),
    "Consensys/gnark#388": ("background_architecture", "ZK proof is library context rather than the change's substantive pattern topic."),
    "Uniswap/web3-react#104": ("insufficient_pattern_discussion", "Transaction plumbing does not establish the selected confirmation mechanism."),
    "trustwallet/assets#2057": ("human_model_disagreement", "Payment detection automation is textual but peripheral to the asset-listing PR topic."),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _split_list(value: Any) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [item for item in text.split("|") if item]


def _group(frame: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frame.fillna("").to_dict("records"):
        grouped[(str(row["repository"]), str(row["issue_number"]))].append(row)
    return grouped


def _run_hashes(path: Path) -> dict[str, str]:
    names = ["run_metadata.json", "stage1_results.csv", "stage2_results.csv", "issue_results.csv"]
    names += [n for n in ["optimization_manifest.json", "compact_catalog.json"] if (path / n).exists()]
    return {name: sha256_file(path / name) for name in names}


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def _load_retrieval(path: Path) -> dict[tuple[str, str], list[str]]:
    manifest = path / "optimization_manifest.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {(str(i["repository"]), str(i["issue_number"])): i.get("retrieval_candidates", []) for i in data.get("issues", [])}


def _load_compact(path: Path, catalog) -> dict[str, dict[str, Any]]:
    compact = path / "compact_catalog.json"
    if compact.exists():
        data = json.loads(compact.read_text(encoding="utf-8"))
        return {item["canonical_name"]: item for item in data.get("patterns", [])}
    return {name: {"aliases": [], "keywords": [], "false_friends": [], "overlap_family": [], "short_definition": catalog.by_name[name].get("description", "")} for name in catalog.names}


def _pair_quality(labels: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame):
    positive = {tuple(row) for row in labels.loc[labels.human_verdict == "yes", PAIR_KEY].itertuples(index=False, name=None)}
    candidates = set(stage2_pairs_from_stage1(s1))
    missed = positive - candidates
    verdicts = {(str(r.repository), str(r.issue_number), str(r.pattern)): str(r.verdict) for r in s2.itertuples() if r.request_status == "succeeded"}
    actual, strict, cond_actual, cond_pred = [], [], [], []
    for row in labels[labels.human_verdict.isin(["yes", "no"])].itertuples():
        key = (str(row.repository), str(row.issue_number), str(row.pattern))
        verdict = verdicts.get(key, "no")
        actual.append(row.human_verdict)
        strict.append("yes" if verdict == "yes" else "no")
        if key in verdicts and verdict in {"yes", "no"}:
            cond_actual.append(row.human_verdict)
            cond_pred.append(verdict)
    conditional = binary_metrics(cond_actual, cond_pred) if cond_actual else None
    end_to_end = binary_metrics(actual, strict) if actual else None
    return {
        "stage1_candidate_recall": (len(positive) - len(missed)) / len(positive) if positive else None,
        "stage1_missed_positives": [list(pair) for pair in sorted(missed)],
        "stage2_conditional": conditional,
        "stage2_conditional_binary": conditional,
        "pair_level_end_to_end": end_to_end,
        "end_to_end_strict_yes": end_to_end,
        "note": "Pair labels contain eight positives and no explicit negative-pair universe; pair precision cannot expose issue-level false positives.",
    }, missed


def _stage_counts(issues: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, retrieval):
    candidate_keys = {(str(r.repository), str(r.issue_number)) for r in s1.itertuples() if r.request_status == "succeeded" and _split_list(r.candidates)}
    sent_keys = {(str(r.repository), str(r.issue_number)) for r in s2.itertuples()}
    yes_keys = {(str(r.repository), str(r.issue_number)) for r in s2.itertuples() if r.request_status == "succeeded" and r.verdict == "yes"}
    final_keys = {(str(r.repository), str(r.issue_number)) for r in issues.itertuples() if r.relevant_to_pattern_study == "yes"}
    return {
        "issues_with_retrieval_shortlist": len([v for v in retrieval.values() if v]) if retrieval else None,
        "issues_with_stage1_candidates": len(candidate_keys),
        "issues_sent_to_stage2": len(sent_keys),
        "issues_with_stage2_yes": len(yes_keys),
        "issues_relevant_after_aggregation": len(final_keys),
        "aggregation_matches_stage2_yes_rule": yes_keys == final_keys,
        "stage2_yes_not_final_yes": [list(x) for x in sorted(yes_keys - final_keys)],
        "final_yes_without_stage2_yes": [list(x) for x in sorted(final_keys - yes_keys)],
    }


def _classification(human: str, predicted: bool) -> str:
    if human == "uncertain": return "EXCLUDED_UNCERTAIN"
    if human == "insufficient_context": return "EXCLUDED_INSUFFICIENT_CONTEXT"
    if human == "yes": return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


def _origin(classification, human_patterns, s1_patterns, s2_rows, aggregation_matches):
    if classification == "FP":
        return "stage2_false_positive" if aggregation_matches else "aggregation_false_positive"
    if classification != "FN": return "not_determinable"
    if not set(human_patterns) & set(s1_patterns): return "stage1_miss"
    relevant = [r for r in s2_rows if r.get("pattern") in human_patterns]
    if relevant and not any(r.get("verdict") == "yes" for r in relevant): return "stage2_false_negative"
    return "aggregation_false_negative" if not aggregation_matches else "not_determinable"


def _diagnosis(issue_key: str, classification: str):
    if classification not in {"FP", "FN"}: return "", ""
    return PILOT_DIAGNOSES.get(issue_key, ("needs_human_review", "Available artifacts do not support a narrower diagnosis."))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _audit_rows(human: pd.DataFrame, run_data):
    rows = []
    for item in human.fillna("").to_dict("records"):
        key = (str(item["repository"]), str(item["issue_number"]))
        issue_key = f"{key[0]}#{key[1]}"
        row = {"repository": key[0], "issue_number": key[1], "human_label": item["human_label"], "human_verdict": item["human_label"], "human_notes": item.get("human_notes", ""), "human_confirmed_patterns": item.get("confirmed_patterns", "")}
        for name, data in run_data.items():
            issue = data["issues_by_key"][key]
            s1 = data["s1_by_key"].get(key, [{}])[0]
            s2_rows = data["s2_by_key"].get(key, [])
            predicted = issue.get("relevant_to_pattern_study") == "yes"
            cls = _classification(item["human_label"], predicted)
            s1_patterns = _split_list(s1.get("candidates", ""))
            try:
                s1_details = json.loads(s1.get("candidate_details", "") or "[]")
            except json.JSONDecodeError:
                s1_details = []
            diagnosis, basis = _diagnosis(issue_key, cls)
            origin = _origin(cls, _split_list(item.get("confirmed_patterns", "")), s1_patterns, s2_rows, data["stage_counts"]["aggregation_matches_stage2_yes_rule"])
            row.update({
                f"{name}_final_verdict": issue.get("relevant_to_pattern_study", ""), f"{name}_relevant": predicted,
                f"{name}_classification": cls, f"{name}_final_patterns": issue.get("patterns_confirmed", ""),
                f"{name}_stage1_candidates": "|".join(s1_patterns), f"{name}_stage1_details": s1.get("candidate_details", ""),
                f"{name}_stage1_evidence": _json([r.get("evidence_text", "") for r in s1_details]),
                f"{name}_stage1_rationale": _json([r.get("rationale", "") for r in s1_details]),
                f"{name}_stage1_confidence": _json([r.get("confidence", "") for r in s1_details]),
                f"{name}_stage2_results": _json(s2_rows),
                f"{name}_stage2_candidate": _json([r.get("pattern", "") for r in s2_rows]),
                f"{name}_stage2_verdict": _json([r.get("verdict", "") for r in s2_rows]),
                f"{name}_stage2_evidence": _json([r.get("evidence_text", "") for r in s2_rows]),
                f"{name}_stage2_rationale": _json([r.get("justification", "") for r in s2_rows]),
                f"{name}_stage2_confidence": _json([r.get("confidence", "") for r in s2_rows]),
                f"{name}_aggregation_result": _json({k: issue.get(k, "") for k in ["relevant_to_pattern_study", "patterns_present", "patterns_confirmed", "patterns_uncertain", "patterns_insufficient_context", "evidence_text", "confidence", "false_friend_detected"]}),
                f"{name}_operational_origin": origin, f"{name}_semantic_diagnosis": diagnosis, f"{name}_diagnosis_basis": basis,
            })
        row["retrieval_shortlist"] = "|".join(run_data["optimized"]["retrieval"].get(key, []))
        # Unprefixed aliases make the main optimized audit convenient while the
        # profile-prefixed columns retain the complete A/B provenance.
        for field in ("stage1_evidence", "stage1_rationale", "stage1_confidence", "stage2_candidate", "stage2_verdict", "stage2_evidence", "stage2_rationale", "stage2_confidence", "aggregation_result", "final_patterns"):
            row[field] = row[f"optimized_{field}"]
        rows.append(row)
    return rows


def _pattern_rows(audit, compact):
    fp, tp, evidence = Counter(), Counter(), defaultdict(list)
    for row in audit:
        target = fp if row["optimized_classification"] == "FP" else tp if row["optimized_classification"] == "TP" else None
        if target is None: continue
        s2 = json.loads(row["optimized_stage2_results"])
        for pattern in _split_list(row["optimized_final_patterns"]):
            target[pattern] += 1
            if target is fp:
                evidence[pattern].extend(r.get("evidence_text", "") for r in s2 if r.get("pattern") == pattern and r.get("verdict") == "yes")
    result = []
    for pattern in sorted(fp, key=lambda p: (-fp[p], p)):
        total, meta = fp[pattern] + tp[pattern], compact.get(pattern, {})
        result.append({"pattern": pattern, "fp_count": fp[pattern], "tp_count": tp[pattern], "total_positive_decisions": total, "observed_fp_proportion": fp[pattern] / total if total else None, "aliases": "|".join(meta.get("aliases", [])), "keywords": "|".join(meta.get("keywords", [])), "false_friends": "|".join(meta.get("false_friends", [])), "family": "|".join(meta.get("overlap_family", [])), "compact_definition": meta.get("short_definition", ""), "llm_evidence_examples": _json(list(dict.fromkeys(evidence[pattern])))})
    return result


def _time_case(run_data, compact, catalog, metadata):
    key, pattern = ("OpenZeppelin/openzeppelin-contracts", "3735"), "Time-Constrained Access"
    legacy = run_data["baseline"]["s1_by_key"][key][0]
    optimized = run_data["optimized"]["s1_by_key"][key][0]
    issue = run_data["optimized"]["issues_by_key"][key]
    manifest_data = json.loads((run_data["optimized"]["path"] / "optimization_manifest.json").read_text())
    manifest_issue = next(i for i in manifest_data["issues"] if (str(i["repository"]), str(i["issue_number"])) == key)
    return {
        "repository": key[0], "issue_number": key[1], "pattern": pattern,
        "optimized_retrieved": pattern in run_data["optimized"]["retrieval"].get(key, []),
        "legacy_stage1_candidate": pattern in _split_list(legacy.get("candidates", "")),
        "optimized_stage1_candidate": pattern in _split_list(optimized.get("candidates", "")),
        "legacy_included_char_count": legacy.get("included_char_count"), "optimized_included_char_count": optimized.get("included_char_count"), "original_char_count": optimized.get("original_char_count"),
        "original_title": issue.get("issue_title", ""), "original_body": issue.get("issue_body", ""), "original_comments": issue.get("concatenated_comments", ""),
        "optimized_selected_spans": manifest_issue.get("selected_spans", {}),
        "legacy_artifact_sha256": next(i["artifact_sha256"] for i in json.loads((run_data["baseline"]["path"] / "request_manifest.json").read_text()) if (str(i["repository"]), str(i["issue_number"])) == key),
        "optimized_artifact_sha256": manifest_issue.get("artifact_sha256"),
        "legacy_stage1_rules_sha256": metadata["baseline"].get("stage1_rules_sha256"), "optimized_stage1_rules_sha256": metadata["optimized"].get("stage1_rules_sha256"),
        "legacy_profile": metadata["baseline"].get("profile"), "optimized_profile": metadata["optimized"].get("profile"),
        "legacy_stage1_candidate_details": legacy.get("candidate_details", ""), "optimized_stage1_candidate_details": optimized.get("candidate_details", ""),
        "legacy_stage2_result": [r for r in run_data["baseline"]["s2_by_key"].get(key, []) if r.get("pattern") == pattern],
        "optimized_stage2_result": [r for r in run_data["optimized"]["s2_by_key"].get(key, []) if r.get("pattern") == pattern],
        "full_definition": catalog.by_name[pattern].get("description", ""), "compact_definition": compact.get(pattern, {}).get("short_definition", ""),
        "hypothesis": "not_determinable",
        "finding": "Optimized retrieval retained the pattern and selected the complete body/comments; compact and full definitions express the same time-window mechanism. The miss occurred inside Stage 1. Prompt/profile and stochastic generation both changed, so prompt_difference cannot be separated from model_variability in one observation.",
    }


def _pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _markdown(report, audit, pattern_rows):
    h = report["human_reference"]
    base = report["runs"]["baseline"]["quality"]["issue_level_binary"]
    opt = report["runs"]["optimized"]["quality"]["issue_level_binary"]
    divergences = [r for r in audit if r["baseline_classification"] in {"FP", "FN"} or r["optimized_classification"] in {"FP", "FN"}]
    opt_fp = [r for r in audit if r["optimized_classification"] == "FP"]
    diag = Counter(r["optimized_semantic_diagnosis"] for r in opt_fp)
    origins = Counter(r["optimized_operational_origin"] for r in opt_fp)
    top = ", ".join(f"{r['pattern']} ({r['fp_count']})" for r in pattern_rows) or "nenhum"
    lines = [
        "# Auditoria issue-level — pilot A/B v2", "", "> Referência: **human-reviewed pilot annotations**; não é gold standard independente.", "",
        "## Síntese objetiva", "",
        f"- Issues: {h['total_issues']} (yes={h['yes']}, no={h['no']}, uncertain={h['uncertain']}, insufficient_context={h['insufficient_context']}); avaliação binária n={h['binary_n']}.",
        f"- Positivas finais: legacy={base['predicted_positive_all_labels']}; optimized={opt['predicted_positive_all_labels']}. O número 18 é final: 7 TP + 10 FP + 1 positiva humana uncertain excluída da matriz.",
        f"- Legacy: TP={base['tp']}, FP={base['fp']}, FN={base['fn']}, TN={base['tn']}; precision={_pct(base['precision'])}, recall={_pct(base['recall'])}, F1={_pct(base['f1'])}, specificity={_pct(base['specificity'])}, FPR={_pct(base['false_positive_rate'])}.",
        f"- Optimized: TP={opt['tp']}, FP={opt['fp']}, FN={opt['fn']}, TN={opt['tn']}; precision={_pct(opt['precision'])}, recall={_pct(opt['recall'])}, F1={_pct(opt['f1'])}, specificity={_pct(opt['specificity'])}, FPR={_pct(opt['false_positive_rate'])}.",
        f"- Divergências binárias únicas: {len(divergences)}. Patterns FP optimized: {top}.", f"- Diagnósticos FP optimized: {dict(diag)}. Origem operacional: {dict(origins)}.", "",
        "A agregação reconstituída confirma que toda e somente issue com pelo menos um Stage 2 `yes` recebeu decisão final `yes`. Não foi encontrado bug de agregação. O comparador v1 avaliava 8 pares humanos positivos e nenhum par negativo explícito, ocultando FP issue-level.", "",
        "## Legacy vs Optimized", "", "| Métrica | Legacy | Optimized | Delta |", "|---|---:|---:|---:|",
    ]
    metrics = [("Predicted relevant issues", "predicted_positive_all_labels"), ("Predicted non-relevant issues (binary)", "predicted_negative"), ("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("TN", "tn"), ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("Specificity", "specificity"), ("F1", "f1"), ("False positive rate", "false_positive_rate"), ("False negative rate", "false_negative_rate"), ("Balanced accuracy", "balanced_accuracy")]
    for label, key in metrics:
        a, b = base[key], opt[key]
        lines.append(f"| {label} | {_pct(a)} | {_pct(b)} | {_pct(b-a)} |" if isinstance(a, float) or isinstance(b, float) else f"| {label} | {a} | {b} | {b-a:+d} |")
    lines += ["", "## Falsos positivos", ""]
    for r in opt_fp:
        lines.append(f"- `{r['repository']}#{r['issue_number']}` — {r['optimized_final_patterns']}; {r['optimized_semantic_diagnosis']} / {r['optimized_operational_origin']}. {r['optimized_diagnosis_basis']}")
    lines += ["", "## Falsos negativos", "", "Optimized: nenhum. Legacy:"]
    for r in audit:
        if r["baseline_classification"] == "FN": lines.append(f"- `{r['repository']}#{r['issue_number']}` — {r['baseline_operational_origin']}; humano: {r['human_confirmed_patterns']}.")
    lines += ["", "## Diagnóstico por estágio", "", "Os 10 FP optimized tiveram candidato Stage 1 e aceite Stage 2; a origem da decisão final divergente é Stage 2. Stage 1 sobregerou esses candidatos, mas não tomou a decisão final.", "", "## Diagnóstico por pattern", ""]
    for r in pattern_rows: lines.append(f"- {r['pattern']}: FP={r['fp_count']}, TP={r['tp_count']}, proporção FP observada={_pct(r['observed_fp_proportion'])}.")
    lines += ["", "## Caso Time-Constrained Access", "", report["time_constrained_access_case"]["finding"], "", "Classificação: `not_determinable`. Retrieval, contexto e equivalência das definições foram confirmados; prompt vs variabilidade não são separáveis neste run.", "",
        "## Limitações", "", "- Apenas 7 positivos humanos; percentuais devem ser lidos com contagens absolutas.", "- As anotações preservam decisões anteriores da LLM e não são gold standard independente/cego.", "- Não existe universo completo de pares humanos negativos.", "- Diagnósticos semânticos são hipóteses auditáveis, não relabeling automático.", "- Uma execução por perfil não separa variação estocástica de efeitos do prompt.", "",
        "## Recomendações", "", "- Adjudicar manualmente os casos `human_model_disagreement`.", "- Testar estabilidade/repetição em experimento separado, especialmente #3735.", "- Investigar fronteiras de mecanismo em conjunto separado, sem derivar regras destes 200 casos.", "- Manter gates issue-level como requisito permanente.", "",
        "## Respostas às perguntas obrigatórias", "", "1. Sim, optimized classificou 18 issues finais como relevantes.", "2. As 18 são pós-agregação, não contagem intermediária.", f"3. Optimized: {opt['predicted_positive_all_labels']}.", f"4. Legacy: {base['predicted_positive_all_labels']}.", f"5. Optimized recuperou {opt['tp']} das 7 issues humanas yes.", f"6. Legacy recuperou {base['tp']} das 7.", f"7–15. Optimized TP/FP/FN/TN={opt['tp']}/{opt['fp']}/{opt['fn']}/{opt['tn']}; precision/recall/F1/specificity/FPR={_pct(opt['precision'])}/{_pct(opt['recall'])}/{_pct(opt['f1'])}/{_pct(opt['specificity'])}/{_pct(opt['false_positive_rate'])}. Legacy={base['tp']}/{base['fp']}/{base['fn']}/{base['tn']}; {_pct(base['precision'])}/{_pct(base['recall'])}/{_pct(base['f1'])}/{_pct(base['specificity'])}/{_pct(base['false_positive_rate'])}.", "16–21. Casos, patterns, evidências, estágios e diagnósticos estão nos CSVs; os FP optimized terminam em Stage 2.", "22. Não foi encontrado bug na aggregation.", "23. Sim: faltava avaliação issue-level, e os 187 negativos não entravam na precision anterior.", "24. Ambos: havia lacuna de avaliação e há 10 decisões finais optimized divergentes da referência.", "25. O miss #3735 ocorreu no Stage 1; contexto/definição não explicam e prompt vs variabilidade é indeterminável.", "26. Optimized continua candidato experimental: recall=100%, precision issue-level=41,18%, ainda não aprovado.", "27. Adjudicar desacordos e avaliar estabilidade/fronteiras em conjunto separado; nenhum ajuste semântico foi aplicado."]
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> int:
    dirs = {name: Path(getattr(args, name + "_run")) for name in ("baseline", "optimized")}
    metadata = {name: json.loads((path / "run_metadata.json").read_text()) for name, path in dirs.items()}
    for key in ("input_sha256", "patterns_sha256", "stage1_model", "stage2_model", "temperature", "seed", "stage1_thinking_level", "stage2_thinking_level", "stage1_max_tokens", "stage2_max_tokens", "limit", "mode"):
        if metadata["baseline"].get(key) != metadata["optimized"].get(key): raise ValueError(f"Incomparable runs: {key} differs")
    data, _ = load_input(dirs["baseline"] / "input_clean_snapshot.csv")
    catalog = load_patterns(dirs["baseline"] / "pattern_catalog_clean_snapshot.csv")
    prepared = [prepare_issue(row, metadata["baseline"]["max_input_chars"]) for _, row in data.iterrows()]
    input_keys = set(zip(data.repository.astype(str), data.issue_number.astype(str)))
    labels = pd.DataFrame(columns=PAIR_KEY + ["human_verdict"])
    human_raw, human_gold = None, None
    if args.human_issues:
        human_gold = load_human_gold(Path(args.human_issues), catalog, Path(args.human_pairs) if args.human_pairs else None)
        human_raw, _ = read_csv_robust(Path(args.human_issues), source="human_issues")
        human_raw = human_issue_reference(human_raw)
        normalized = human_gold.dataframe.set_index(ISSUE_KEY)
        human_raw["confirmed_patterns"] = [normalized.loc[(r, n), "confirmed_patterns"] for r, n in human_raw[ISSUE_KEY].itertuples(index=False, name=None)]
        labels = pd.DataFrame([{"repository": r, "issue_number": n, "pattern": p, "human_verdict": "yes"} for r, n, p in human_gold.strict_pairs] + [{"repository": r, "issue_number": n, "pattern": p, "human_verdict": "uncertain"} for r, n, p in human_gold.uncertain_pairs])
        if set(map(tuple, human_raw[ISSUE_KEY].itertuples(index=False, name=None))) != input_keys: raise ValueError("Human issue keys and run input keys differ")
    elif args.human_pairs:
        raw, _ = read_csv_robust(Path(args.human_pairs), source="human_pairs")
        labels = raw[PAIR_KEY + ["human_verdict"]].copy()
    if not labels.empty:
        labels = labels.loc[[(str(r), str(n)) in input_keys for r, n in labels[["repository", "issue_number"]].itertuples(index=False, name=None)]]

    report = {"comparison": "live_runs_offline_audit_v2", "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(), "current_git_commit": _git_commit(), "human_positive_pairs": int((labels.human_verdict == "yes").sum()) if not labels.empty else 0, "runs": {}}
    run_data, misses = {}, {}
    for name, path in dirs.items():
        s1 = ResultStore(path / "stage1_results.csv", stage="stage1").dataframe
        s2 = ResultStore(path / "stage2_results.csv", stage="stage2").dataframe
        issues, _ = read_csv_robust(path / "issue_results.csv", source=f"{name}_issue_results")
        issues["issue_number"] = issues["issue_number"].astype(str).str.replace(r"\.0+$", "", regex=True)
        retrieval = _load_retrieval(path)
        integrity = pipeline_integrity_report(prepared, s1, s2)
        quality, misses[name] = _pair_quality(labels, s1, s2)
        counts = _stage_counts(issues, s1, s2, retrieval)
        quality["issue_level_binary"] = issue_level_binary(human_raw, issues)[0] if human_raw is not None else None
        positive = {tuple(x) for x in labels.loc[labels.human_verdict == "yes", PAIR_KEY].itertuples(index=False, name=None)}
        retrieval_recall = retrieval_misses = None
        if (path / "optimization_manifest.json").exists() and positive:
            retrieved = {(r, n, p) for (r, n), patterns in retrieval.items() for p in patterns}
            retrieval_misses = [list(pair) for pair in sorted(positive - retrieved)]
            retrieval_recall = (len(positive) - len(retrieval_misses)) / len(positive)
        quality["retrieval"] = {"pair_recall": retrieval_recall, "missed_positive_pairs": retrieval_misses}
        if integrity["status"] != "OK":
            for field in ["stage1_candidate_recall", "stage2_conditional", "stage2_conditional_binary", "pair_level_end_to_end", "end_to_end_strict_yes", "issue_level_binary"]: quality[field] = None
        token_path = path / "token_summary.json"
        tokens = json.loads(token_path.read_text()) if token_path.exists() else {}
        report["runs"][name] = {"run_id": path.name, "path": str(path), "profile": metadata[name].get("profile"), "commit_hash": metadata[name].get("commit_hash"), "models": {"stage1": metadata[name].get("stage1_model"), "stage2": metadata[name].get("stage2_model")}, "configuration": metadata[name].get("optimization_config"), "artifact_sha256": _run_hashes(path), "integrity": integrity, "pipeline_stage_counts": counts, "tokens": tokens, "quality": quality, "retrieval_recall": retrieval_recall, "retrieval_missed_positives": retrieval_misses}
        grouped_issues = _group(issues)
        run_data[name] = {"path": path, "s1": s1, "s2": s2, "issues": issues, "retrieval": retrieval, "stage_counts": counts, "s1_by_key": _group(s1), "s2_by_key": _group(s2), "issues_by_key": {k: v[0] for k, v in grouped_issues.items()}}

    old, new = [report["runs"][n]["tokens"] for n in ("baseline", "optimized")]
    usage_ok = old.get("usage_complete") and new.get("usage_complete")
    savings = 100 * (old["total_tokens"] - new["total_tokens"]) / old["total_tokens"] if usage_ok and old.get("total_tokens") else None
    report["tokens_saved_vs_baseline"] = old["total_tokens"] - new["total_tokens"] if usage_ok else None
    report["percentage_saved_vs_baseline"] = savings
    report["new_stage1_false_negatives"] = [list(p) for p in sorted(misses["optimized"] - misses["baseline"])] if not labels.empty else None
    if human_gold is not None:
        counts = human_raw.human_label.value_counts().to_dict()
        report["human_reference"] = {"description": "human-reviewed pilot annotations", "independent_gold_standard": False, "path": str(Path(args.human_issues)), "sha256": sha256_file(Path(args.human_issues)), "total_issues": len(human_raw), "yes": int(counts.get("yes", 0)), "no": int(counts.get("no", 0)), "uncertain": int(counts.get("uncertain", 0)), "insufficient_context": int(counts.get("insufficient_context", 0)), "binary_n": int(counts.get("yes", 0) + counts.get("no", 0)), "adapter_summary": human_gold.summary}

    gates = {"complete_runs": all(r["integrity"]["status"] == "OK" for r in report["runs"].values()), "measured_token_reduction_ge_60_percent": savings >= 60 if savings is not None else NOT_MEASURED, "retrieval_recall_ge_98_percent": report["runs"]["optimized"]["retrieval_recall"] >= .98 if report["runs"]["optimized"]["retrieval_recall"] is not None else NOT_MEASURED}
    recalls = [report["runs"][n]["quality"]["stage1_candidate_recall"] for n in ("baseline", "optimized")]
    gates["stage1_recall_drop_le_1pp"] = recalls[0] - recalls[1] <= .01 if all(v is not None for v in recalls) else NOT_MEASURED
    for unit, field in [("pair_level", "pair_level_end_to_end"), ("stage2_conditional", "stage2_conditional")]:
        for metric in ("precision", "recall", "f1"):
            values = [(report["runs"][n]["quality"].get(field) or {}).get(metric) for n in ("baseline", "optimized")]
            gates[f"{unit}_{metric}_drop_le_2pp"] = values[0] - values[1] <= .02 if all(v is not None for v in values) else NOT_MEASURED
            if unit == "pair_level":
                # Compatibility alias used by v1 consumers. Its meaning remains
                # the old strict end-to-end pair metric.
                gates[f"end_to_end_{metric}_drop_le_2pp"] = gates[f"{unit}_{metric}_drop_le_2pp"]
    issue_values = [report["runs"][n]["quality"]["issue_level_binary"] for n in ("baseline", "optimized")]
    gates["issue_level_metrics_measured"] = all(v is not None for v in issue_values)
    for metric in ("precision", "recall", "f1"):
        values = [(v or {}).get(metric) for v in issue_values]
        gates[f"issue_level_{metric}_drop_le_2pp"] = values[0] - values[1] <= .02 if all(v is not None for v in values) else NOT_MEASURED
    gates["issue_level_false_positives_reported"] = human_raw is not None
    report["optimized_issue_level_false_positive_count"] = (issue_values[1] or {}).get("fp")
    report["optimized_issue_level_false_negative_count"] = (issue_values[1] or {}).get("fn")
    report["acceptance_gates"] = gates
    report["acceptance"] = "PASSED_ON_OBSERVED_SAMPLE_ONLY" if all(v is True for v in gates.values()) else "NOT_APPROVED"
    report["statistical_note"] = "Absolute counts are primary; seven human-positive issues do not establish population performance."
    output = Path(args.output)
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    if human_raw is not None:
        compact = _load_compact(dirs["optimized"], catalog)
        audit = _audit_rows(human_raw, run_data)
        pattern_rows = _pattern_rows(audit, compact)
        report["time_constrained_access_case"] = _time_case(run_data, compact, catalog, metadata)
        predictions = [{k: v for k, v in row.items() if not k.endswith("_stage2_results") and not k.endswith("_stage1_details") and not k.endswith("_aggregation_result")} for row in audit]
        _write_csv(output.parent / "issue_level_predictions.csv", predictions)
        _write_csv(output.parent / "issue_level_divergences.csv", [r for r in audit if r["baseline_classification"] in {"FP", "FN"} or r["optimized_classification"] in {"FP", "FN"}])
        positives = []
        for row in audit:
            if not row["optimized_relevant"]: continue
            s2 = json.loads(row["optimized_stage2_results"])
            positives.append({"repository": row["repository"], "issue_number": row["issue_number"], "human_label": row["human_label"], "optimized_relevant": True, "classification": row["optimized_classification"], "predicted_patterns": row["optimized_final_patterns"], "stage1_candidates": row["optimized_stage1_candidates"], "stage2_yes_patterns": "|".join(r["pattern"] for r in s2 if r.get("verdict") == "yes"), "stage2_non_yes_patterns": "|".join(r["pattern"] for r in s2 if r.get("verdict") != "yes"), "final_aggregated_patterns": row["optimized_final_patterns"]})
        order = {"FP": 0, "FN": 1, "TP": 2, "EXCLUDED_UNCERTAIN": 3, "EXCLUDED_INSUFFICIENT_CONTEXT": 4, "TN": 5}
        positives.sort(key=lambda r: (order[r["classification"]], r["repository"], r["issue_number"]))
        _write_csv(output.parent / "optimized_positive_issues.csv", positives)
        error_columns = ["profile", "repository", "issue_number", "classification", "human_label", "final_patterns", "operational_origin", "semantic_diagnosis", "diagnosis_basis", "human_notes", "stage1_candidates", "stage1_details", "stage2_results", "aggregation_result"]
        fps, fns = [], []
        for profile in ("baseline", "optimized"):
            for row in audit:
                cls = row[f"{profile}_classification"]
                if cls not in {"FP", "FN"}: continue
                record = {"profile": profile, "repository": row["repository"], "issue_number": row["issue_number"], "classification": cls, "human_label": row["human_label"], "final_patterns": row[f"{profile}_final_patterns"], "operational_origin": row[f"{profile}_operational_origin"], "semantic_diagnosis": row[f"{profile}_semantic_diagnosis"], "diagnosis_basis": row[f"{profile}_diagnosis_basis"], "human_notes": row["human_notes"], "stage1_candidates": row[f"{profile}_stage1_candidates"], "stage1_details": row[f"{profile}_stage1_details"], "stage2_results": row[f"{profile}_stage2_results"], "aggregation_result": row[f"{profile}_aggregation_result"]}
                (fps if cls == "FP" else fns).append(record)
        _write_csv(output.parent / "false_positives.csv", fps, error_columns)
        _write_csv(output.parent / "false_negatives.csv", fns, error_columns)
        _write_csv(output.parent / "pattern_fp_summary.csv", pattern_rows)
        diagnostic = {"optimized_false_positive_semantic_categories": dict(Counter(r["optimized_semantic_diagnosis"] for r in audit if r["optimized_classification"] == "FP")), "optimized_false_positive_operational_origins": dict(Counter(r["optimized_operational_origin"] for r in audit if r["optimized_classification"] == "FP")), "optimized_false_negative_operational_origins": dict(Counter(r["optimized_operational_origin"] for r in audit if r["optimized_classification"] == "FN")), "pattern_false_positives": pattern_rows, "aggregation_bug_found": not all(r["pipeline_stage_counts"]["aggregation_matches_stage2_yes_rule"] for r in report["runs"].values()), "comparator_v1_issue": "No issue-level universe; pair labels contained eight positives and no explicit negative pairs.", "diagnoses_are_annotations_only": True}
        write_json(output.parent / "diagnostic_summary.json", diagnostic)
        (output.parent / "REPORT.md").write_text(_markdown(report, audit, pattern_rows), encoding="utf-8")
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["acceptance"] == "PASSED_ON_OBSERVED_SAMPLE_ONLY" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--optimized-run", required=True)
    parser.add_argument("--human-issues")
    parser.add_argument("--human-pairs")
    parser.add_argument("--output", required=True)
    return compare(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
