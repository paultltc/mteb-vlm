from __future__ import annotations

import ast

import datasets
import numpy as np

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class IndonesianMongabayConservationClassification(AbsTaskClassification):
    superseded_by = "IndonesianMongabayConservationClassification.v2"
    metadata = TaskMetadata(
        name="IndonesianMongabayConservationClassification",
        description="Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)",
        reference="https://aclanthology.org/2023.sealp-1.4/",
        dataset={
            "path": "Datasaur/mongabay-experiment",
            "revision": "c9e9f2c09836bfec57c543ab65983f3398e9657a",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2012-01-01", "2023-12-31"),
        eval_splits=["validation", "test"],
        eval_langs=["ind-Latn"],
        main_score="f1",
        domains=["Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fransiska-etal-2023-utilizing,
  address = {Nusa Dua, Bali, Indonesia},
  author = {Fransiska, Mega  and
Pitaloka, Diah  and
Saripudin, Saripudin  and
Putra, Satrio  and
Sutawika*, Lintang},
  booktitle = {Proceedings of the First Workshop in South East Asian Language Processing},
  doi = {10.18653/v1/2023.sealp-1.4},
  editor = {Wijaya, Derry  and
Aji, Alham Fikri  and
Vania, Clara  and
Winata, Genta Indra  and
Purwarianti, Ayu},
  month = nov,
  pages = {30--54},
  publisher = {Association for Computational Linguistics},
  title = {Utilizing Weak Supervision to Generate {I}ndonesian Conservation Datasets},
  url = {https://aclanthology.org/2023.sealp-1.4},
  year = {2023},
}
""",
    )

    def dataset_transform(self):
        splits = self.metadata_dict["eval_splits"]
        class_labels = ["positif", "netral", "negatif"]

        ds = {}

        # Include training because the classification task requires it
        train_split = self.dataset["train"]
        train_docs: list = []
        train_labels: list = []
        for text, label in zip(train_split["text"], train_split["softlabel"]):
            soft_label = ast.literal_eval(label)
            if len(soft_label) == len(class_labels):
                train_docs.append(text)
                hard_label = np.argmax(soft_label)
                train_labels.append(hard_label)

        ds["train"] = datasets.Dataset.from_dict(
            {
                "text": train_docs,
                "label": train_labels,
            }
        )

        documents: list = []
        labels: list = []
        # For evaluation
        for split in splits:
            ds_split = self.dataset[split]
            for text, label in zip(ds_split["text"], ds_split["softlabel"]):
                if label in class_labels:
                    documents.append(text)
                    labels.append(class_labels.index(label))

            assert len(documents) == len(labels)

            ds[split] = datasets.Dataset.from_dict(
                {
                    "text": documents,
                    "label": labels,
                }
            )

        self.dataset = datasets.DatasetDict(ds)


class IndonesianMongabayConservationClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="IndonesianMongabayConservationClassification.v2",
        description="""Conservation dataset that was collected from mongabay.co.id contains topic-classification task (multi-label format) and sentiment classification. This task only covers sentiment analysis (positive, neutral negative)
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://aclanthology.org/2023.sealp-1.4/",
        dataset={
            "path": "mteb/indonesian_mongabay_conservation",
            "revision": "04863a3b6885470071f649a4d4dcd7e9d8e98cf8",
        },
        type="Classification",
        category="s2s",
        modalities=["text"],
        date=("2012-01-01", "2023-12-31"),
        eval_splits=["validation", "test"],
        eval_langs=["ind-Latn"],
        main_score="f1",
        domains=["Web", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{fransiska-etal-2023-utilizing,
  address = {Nusa Dua, Bali, Indonesia},
  author = {Fransiska, Mega  and
Pitaloka, Diah  and
Saripudin, Saripudin  and
Putra, Satrio  and
Sutawika*, Lintang},
  booktitle = {Proceedings of the First Workshop in South East Asian Language Processing},
  doi = {10.18653/v1/2023.sealp-1.4},
  editor = {Wijaya, Derry  and
Aji, Alham Fikri  and
Vania, Clara  and
Winata, Genta Indra  and
Purwarianti, Ayu},
  month = nov,
  pages = {30--54},
  publisher = {Association for Computational Linguistics},
  title = {Utilizing Weak Supervision to Generate {I}ndonesian Conservation Datasets},
  url = {https://aclanthology.org/2023.sealp-1.4},
  year = {2023},
}
""",
        adapted_from=["IndonesianMongabayConservationClassification"],
    )
