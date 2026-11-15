import json

scores_path = r"d:\Desktop\GitHub\AI Banking Asistant\dataset\agent_eval_scores_785_cache.json"
answers_path = r"d:\Desktop\GitHub\AI Banking Asistant\dataset\agent_answers_785_cache.json"

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

print("\n--- TOP 10 FAILED SAMPLES ---")
for k in failed[:10]:
    sc = scores[k]
    ans = answers.get(k, {}).get("answer", "")
    print(f"Case {k}: Acc={sc.get('accuracy')}, Faith={sc.get('faithfulness')}, Comp={sc.get('completeness')}")
    print(f"Feedback: {sc.get('feedback')}")
    print(f"Answer:\n{ans[:400]}")
    print("=" * 60)
