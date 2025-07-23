from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.Image.AbsTaskAny2AnyRetrieval import AbsTaskAny2AnyRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

def _load_data(
    path: str,
    splits: str,
    cache_dir: str | None = None,
    revision: str | None = None,
):
    queries = {}
    corpus = {}
    relevant_docs = {}

    for split in splits:
        ds = load_dataset(
            path,
            split=split,
            cache_dir=cache_dir,
            revision=revision,
        )
        ds = ds.select_columns(["caption", "image"])

        ds = ds.map(
            lambda x, i: {
                "query-id": f"query-{split}-{i}",
                "query": x["image"],
                "corpus-id": f"corpus-{split}-{i}",
                "corpus": x["caption"],
            },
            with_indices=True,
            remove_columns=["caption", "image"],
        )

        query_ds = ds.map(
            lambda x: {
                "id": x["query-id"],
                "text": None,
                "image": x["query"],
                "modality": "image",
            },
            remove_columns=["query-id", "query"],
        )

        corpus_ds = ds.map(
            lambda x: {
                "id": x["corpus-id"],
                "text": x["corpus"],
                "image": None,
                "modality": "text",
            },
            remove_columns=["corpus-id", "corpus"],
        )

        qrels_ds = ds.map(
            lambda x: {
                "query-id": x["query-id"],
                "corpus-id": x["corpus-id"],
                "score": 1,
            },
            remove_columns=["query-id", "corpus-id"],
        )

        queries[split] = query_ds
        corpus[split] = corpus_ds
        relevant_docs[split] = {}
        for row in qrels_ds:
            qid = row["query-id"]
            did = row["corpus-id"]
            if qid not in relevant_docs[split]:
                relevant_docs[split][qid] = {}
            relevant_docs[split][qid][did] = int(row["score"])

    return corpus, queries, relevant_docs

class NatCapCarsI2TRetrieval(AbsTaskAny2AnyRetrieval):
    metadata = TaskMetadata(
        name="NatCapCarsI2TRetrieval",
        description="...",
        reference="https://huggingface.co/datasets/SmolVEncoder/natcap",
        dataset={
            "path": "SmolVEncoder/natcap",
            "name": "cars",
            "revision": "79cb752f05ea04b6e59211c5d414d86f7fba8108",
            "trust_remote_code": True,
        },
        type="Any2AnyRetrieval",
        category="i2t",
        eval_splits=["val"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_5",
        date=("2025-07-01", "2025-07-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Image Text Retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        modalities=["text", "image"],
        sample_creation="found",
        bibtex_citation=r"""...""",
        prompt={
            "query": "Find an image caption describing the following everyday image."
        },
        descriptive_stats={
            "n_samples": {"val": 30000},
            "avg_character_length": {
                "val": {
                    "average_document_length": 30.94235294117647,
                    "average_query_length": 131.56569965870307,
                    "num_documents": 24809,
                    "num_queries": 30000,
                    "average_relevant_docs_per_query": 1.0,
                }
            },
        },
    )

    def load_data(self, **kwargs):
            if self.data_loaded:
                return

            self.corpus, self.queries, self.relevant_docs = _load_data(
                path=self.metadata_dict["dataset"]["path"],
                splits=self.metadata_dict["eval_splits"],
                cache_dir=kwargs.get("cache_dir", None),
                revision=self.metadata_dict["dataset"]["revision"],
            )

            self.data_loaded = True