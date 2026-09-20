"""Prompt and rubric text for arms C and D, versioned by CONFIG.PROMPT_VERSION.

Isolated in its own module because this text IS the experiment: it is the most
consequential and most arbitrary thing in the repo, and every change to it
requires human review and a prompt_version bump.

The two arms' rubrics hold each other to CLAUDE.md non-negotiable #4 (equal
effort for every arm): concrete self-contained levels, and an explicit ruling on
the case a careless grader gets wrong - a document that refutes the query's
claim is still relevant to deciding it.
"""

LLM_RERANK_PROMPT = """\
You are grading how well a document supports answering a search query.

QUERY: {query}

DOCUMENT: {document}

Grade the document on this scale:
0 - The document is about an unrelated subject.
1 - The document is in the same broad field but does not address the query.
2 - The document addresses the query's subject and contains partial or indirect
    evidence bearing on it.
3 - The document contains evidence that settles the query, either by supporting
    it or by contradicting it.

A document that contradicts the query's claim with evidence is a 3, not a 0 —
it is directly relevant to deciding the claim.

Reply with the single digit and nothing else."""


JEV_STATE = """\
QUERY: {query}

DOCUMENT: {document}"""

# Levels describe SITUATIONS a document either is or is not in, never a degree
# on a low/medium/high scale. Two reasons, both from TypeSafe's own Score
# guidance: (1) a vague degree word invites the literal-reading failure mode
# they document, where the model anchors on the adjective instead of the
# document; (2) a level must be judgeable on its own, without reading its
# neighbours, or an unordered read of the rubric changes the verdict.
#
# `cookbook_relevant` is a fourth question, added on top of BRD §4.4's three
# pre-registered ones (orchestrator addition, approved 2026-09-20): it runs
# TypeSafe's own reranking-cookbook recipe (docs.typesafe.ai/cookbooks/
# rerank_typesafe.md — one Noul per candidate, no composition) inside the SAME
# call/state as the composed arm, so the two designs can be compared without
# doubling API spend. `compose_relevance` below must never read it.
JEV_QUESTIONS_V1 = {
    "topical_overlap": {
        "type": "score",
        "instructions": "How closely does the document's subject match the query?",
        "criteria": [
            "The document is about an unrelated subject.",
            "The document is in the same broad field but concerns a different "
            "subject than the query.",
            "The document concerns the query's specific subject but does not "
            "address the claim the query is making about it.",
            "The document is directly about the claim the query is making.",
        ],
    },
    "answers_query": {
        "type": "score",
        "instructions": "How well does the document let a reader decide the query's claim?",
        "criteria": [
            "The document contains nothing bearing on the claim.",
            "The document contains partial or indirect evidence bearing on the claim.",
            "The document contains evidence that settles the claim, either by "
            "supporting it or by contradicting it.",
        ],
    },
    "is_contradictory": {
        "type": "noul",
        "instructions": "This document presents evidence against the claim made in the query.",
    },
    "cookbook_relevant": {
        "type": "noul",
        "instructions": (
            "This document, by itself, contains what a reader needs to answer the query."
        ),
        # The cookbook's recipe states both conditions explicitly rather than
        # leaving the negative case implied. Verified against the SDK: Noul
        # takes criteria={"true": ..., "false": ...}. Omitting them would run a
        # weaker version of TypeSafe's own recommendation than the one they
        # document, which under the equal-effort rule is a bias in favour of
        # the composed arm this is meant to check.
        "criteria": {
            "true": (
                "The document states the specific fact, finding, or rule that the "
                "query asks about, so a reader could decide the query from this "
                "document alone."
            ),
            "false": (
                "The document is on a similar topic but does not supply the "
                "specific proposition the query turns on; a reader would still "
                "need another document."
            ),
        },
    },
}
