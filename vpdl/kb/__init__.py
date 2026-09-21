"""Local clinical-genetics knowledge base: retrieve, cite, or say "not found".

Design and rules: ``docs/kb/DESIGN.md``. How to run it: ``docs/kb/RUNBOOK.md``.

    genereviews.py   GeneReviews chapters (NCBI Bookshelf XML) -> cited passages
    search.py        exact-word (BM25) + meaning (embedding) search, merged
    ollama.py        the local model server; refuses to talk to any other machine
    variants.py      ClinVar lookups — variant facts come from here, never a model
    answer.py        question -> passages -> local model -> citation check
    evaluate.py      the question set that decides which model is trusted

The model's only job is to summarise retrieved passages. Knowledge lives in the
documents; an answer that does not cite them is withheld.
"""
