"""Terminal RAG application with baseline, improved, and comparison modes."""

import argparse
import math

from src.ingest.enterprise_ingest import ingest_documents
from src.rag.pipelines import compare_question, format_result, run_pipeline
from src.search.scored_retriever import get_search_store


def initialize_knowledge_base():
    """Refresh the active index; incremental ingestion skips unchanged PDFs."""
    print("Checking PDFs and the active embedding/chunking configuration...")
    ingest_documents()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "improved", "compare"], default="baseline")
    parser.add_argument("--question", help="Answer one question and exit; otherwise open the question loop.")
    parser.add_argument("--top-k", type=int, default=5, help="Candidate count for both pipelines.")
    parser.add_argument("--source", help="Exact source filename; improved mode only.")
    parser.add_argument("--category", help="Exact category; improved mode only.")
    parser.add_argument("--max-distance", type=float, help="Optional raw distance cutoff for improved mode; lower is closer. Calibrate before use.")
    parser.add_argument("--show-context", action="store_true", help="Print full retrieved passages, including rejected candidates.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.max_distance is not None and not math.isfinite(args.max_distance):
        parser.error("--max-distance must be finite")
    for name in ("question", "source", "category"):
        value = getattr(args, name)
        if value is not None and not value.strip():
            parser.error(f"--{name} cannot be blank")
    if args.mode == "baseline" and any(value is not None for value in (args.source, args.category, args.max_distance)):
        parser.error("--source, --category, and --max-distance require improved or compare mode")
    return args


def main(argv=None):
    args = parse_args(argv)
    initialize_knowledge_base()
    # Load common search resources before either pipeline's retrieval timer.
    store = get_search_store()
    print(f"\nSoftware Knowledge AI | mode: {args.mode}")
    if args.mode == "compare":
        print("Baseline uses unfiltered top-k. Filters/cutoff apply only to improved.")
        print("Comparison can make two LLM calls; timings are diagnostic, not a benchmark.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = args.question if args.question is not None else input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ["exit", "quit"]:
            break
        if not question.strip():
            continue
        options = dict(top_k=args.top_k, source=args.source, category=args.category,
                       max_distance=args.max_distance, store=store)
        if args.mode == "compare":
            results = compare_question(question, **options)
        else:
            results = [run_pipeline(question, mode=args.mode, **options)]
        for result in results:
            print(format_result(result, show_context=args.show_context))
        if args.question is not None:
            break


if __name__ == "__main__":
    main()
