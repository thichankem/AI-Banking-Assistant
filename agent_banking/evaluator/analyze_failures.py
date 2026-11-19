import json
from pathlib import Path

from agent_banking.evaluator.config.settings import EvaluatorConfig, resolve_repo_path

_OUT_CFG = EvaluatorConfig.CACHES_AND_OUTPUTS
_J785 = EvaluatorConfig.JUDGE_785


def main():
    root = Path(__file__).resolve().parents[2]
    scores_path = resolve_repo_path(_OUT_CFG["scores_785_cache_relpath"], root=root)
    answers_path = resolve_repo_path(_OUT_CFG["answers_785_cache_relpath"], root=root)

    with open(scores_path, "r", encoding="utf-8") as f:
        scores = json.load(f)

    with open(answers_path, "r", encoding="utf-8") as f:
        answers = json.load(f)

    print(f"Total scored: {len(scores)}")
    passed = []
    failed = []
    for k, v in scores.items():
        if v.get("verdict") == "PASS":
            passed.append(k)
        else:
            failed.append(k)

    print(f"PASS: {len(passed)}, FAIL: {len(failed)}")

    top_n = int(_J785["top_failed_samples"])
    preview_chars = int(_J785["answer_preview_chars"])
    print(f"\n--- TOP {top_n} FAILED SAMPLES ---")
    for k in failed[:top_n]:
        sc = scores[k]
        ans = answers.get(k, {}).get("answer", "")
        print(f"Case {k}: Acc={sc.get('accuracy')}, Faith={sc.get('faithfulness')}, Comp={sc.get('completeness')}")
        print(f"Feedback: {sc.get('feedback')}")
        print(f"Answer:\n{ans[:preview_chars]}")
        print("=" * 60)


if __name__ == "__main__":
    main()
