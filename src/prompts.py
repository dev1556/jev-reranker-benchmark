"""Prompt and rubric text for arms C and D, versioned by CONFIG.PROMPT_VERSION.

Isolated in its own module because this text IS the experiment: it is the most
consequential and most arbitrary thing in the repo, and every change to it
requires human review and a prompt_version bump.
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
