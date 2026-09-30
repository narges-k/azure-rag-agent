"""Step 4: measure quality. Never judge a RAG system by 'it looked good on 3 questions'.

1) Retrieval: for each test question we know which document should be found -> hit@k and MRR.
2) Generation (optional --judge): an LLM grades whether the answer is grounded in the retrieved sources.

Create eval_questions.json like:
[
  {"question": "dofference between diagnostic mode and chatmode?", "expected_source": "CTH_RFP.pdf"},
  {"question": "How much is the grade of customer approva", "expected_source": "Project_Memo_2026.pdf"}
]
Usage: python evaluate.py --k 5 [--semantic] [--judge]
"""
import argparse
import json

from ask import answer, build_context, retrieve
from common import env, openai_client

JUDGE_PROMPT = """You are a strict grader. Given SOURCES, a QUESTION and an ANSWER, rate whether every claim in the
ANSWER is supported by the SOURCES. Reply with JSON only: {"grounded": 1-5, "reason": "short"}.
5 = fully supported, 1 = mostly unsupported. If the answer says it does not know and the sources really do not
contain the answer, give 5."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--file", default="eval_questions.json")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        cases = json.load(f)

    hits_at_k, reciprocal_ranks, grounded_scores, answerable = 0, 0.0, [], 0
    client = openai_client() if args.judge else None
    for case in cases:
        hits = retrieve(case["question"], args.k, args.semantic)
        expected = case.get("expected_source")
        if expected:
            answerable += 1
            rank = next((i for i, h in enumerate(hits, 1) if h["source"] == expected), None)
            if rank:
                hits_at_k += 1
                reciprocal_ranks += 1 / rank
            print(f"{'HIT ' if rank else 'MISS'} rank={rank} | {case['question']}")
        if args.judge:
            text, used = answer(case["question"], args.k, args.semantic)
            resp = client.chat.completions.create(
                model=env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"SOURCES:\n{build_context(used)}\n\nQUESTION: {case['question']}\n\nANSWER: {text}"},
                ],
            )
            try:
                grounded_scores.append(json.loads(resp.choices[0].message.content)["grounded"])
            except (ValueError, KeyError, TypeError):
                print("  (judge output could not be parsed)")

    if answerable:
        print(f"\nRetrieval hit@{args.k}: {hits_at_k}/{answerable} = {hits_at_k / answerable:.0%}")
        print(f"MRR: {reciprocal_ranks / answerable:.2f}")
    if grounded_scores:
        print(f"Groundedness (LLM judge, 1-5): {sum(grounded_scores) / len(grounded_scores):.2f} over {len(grounded_scores)} answers")


if __name__ == "__main__":
    main()
