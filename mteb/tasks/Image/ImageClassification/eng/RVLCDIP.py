from __future__ import annotations

from mteb.abstasks.Image.AbsTaskImageClassification import AbsTaskImageClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RVLCDIPClassification(AbsTaskImageClassification):
    metadata = TaskMetadata(
        name="RVLCDIP",
        description="RVLCDIP is a large-scale dataset for document image classification, containing a wide variety of document types.",
        reference="https://arxiv.org/pdf/1502.07058",
        dataset={
            "path": "aharley/rvl_cdip",
            "revision": "03f14a4ad0a32413eff51ca10f9f511545f2bd5b",
        },
        type="ImageClassification",
        category="i2i",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2015-01-01",
            "2015-12-01",
        ),  # Estimated range for the collection of reviews
        domains=["Encyclopaedic"],
        task_subtypes=["Document classification"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        modalities=["image"],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})},
    year = {2015}
}
""",
        descriptive_stats={
            "n_samples": {"test": 40000},
            "avg_character_length": {"test": 0},
        },
    )
    image_column_name: str = "image"
    label_column_name: str = "label"
