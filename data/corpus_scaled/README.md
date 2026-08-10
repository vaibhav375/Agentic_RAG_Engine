# Corpus

This directory holds the source documents that get ingested.

The bundled files (`01_routing.md` … `10_configuration.md`) are an **original,
self-contained technical-documentation corpus** for a fictional Python web
framework ("Breeze"). They are authored specifically so that:

- every gold answer is verifiable against a source passage,
- there is no copyright or network dependency, and
- the whole pipeline (ingest → retrieve → generate → eval) runs offline.

To run against **real** docs instead, use `python data/download_corpus.py`
(see its `--help`) to pull a public framework's markdown into
`data/corpus/downloaded/`, point `corpus_dir` at it, and build a new
`data/eval/gold_qa.jsonl` for that domain. The gold set is corpus-specific.
