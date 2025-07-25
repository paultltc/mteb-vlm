from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.TaskMetadata import TaskMetadata


class RuReviewsClassification(AbsTaskClassification):
    superseded_by = "RuReviewsClassification.v2"
    metadata = TaskMetadata(
        name="RuReviewsClassification",
        dataset={
            "path": "ai-forever/ru-reviews-classification",
            "revision": "f6d2c31f4dc6b88f468552750bfec05b4b41b05a",
        },
        description="Product review classification (3-point scale) based on RuRevies dataset",
        reference="https://github.com/sismetanin/rureviews",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2000-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Smetanin-SA-2019,
  author = {Sergey Smetanin and Michail Komarov},
  booktitle = {2019 IEEE 21st Conference on Business Informatics (CBI)},
  doi = {10.1109/CBI.2019.00062},
  issn = {2378-1963},
  month = {July},
  number = {},
  pages = {482-486},
  title = {Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks},
  volume = {01},
  year = {2019},
}
""",
        prompt="Classify product reviews into positive, negative or neutral sentiment",
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )


class RuReviewsClassificationV2(AbsTaskClassification):
    metadata = TaskMetadata(
        name="RuReviewsClassification.v2",
        dataset={
            "path": "mteb/ru_reviews",
            "revision": "46d80ee5ac51be8234725558677e59050b9c418e",
        },
        description="""Product review classification (3-point scale) based on RuRevies dataset
        This version corrects errors found in the original data. For details, see [pull request](https://github.com/embeddings-benchmark/mteb/pull/2900)""",
        reference="https://github.com/sismetanin/rureviews",
        type="Classification",
        category="p2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="accuracy",
        date=("2000-01-01", "2020-01-01"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Smetanin-SA-2019,
  author = {Sergey Smetanin and Michail Komarov},
  booktitle = {2019 IEEE 21st Conference on Business Informatics (CBI)},
  doi = {10.1109/CBI.2019.00062},
  issn = {2378-1963},
  month = {July},
  number = {},
  pages = {482-486},
  title = {Sentiment Analysis of Product Reviews in Russian using Convolutional Neural Networks},
  volume = {01},
  year = {2019},
}
""",
        prompt="Classify product reviews into positive, negative or neutral sentiment",
        adapted_from=["RuReviewsClassification"],
    )

    def dataset_transform(self):
        self.dataset = self.stratified_subsampling(
            self.dataset, seed=self.seed, n_samples=2048, splits=["test"]
        )
