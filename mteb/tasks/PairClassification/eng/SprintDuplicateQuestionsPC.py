from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


class SprintDuplicateQuestionsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="SprintDuplicateQuestions",
        description="Duplicate questions from the Sprint community.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "mteb/sprintduplicatequestions-pairclassification",
            "revision": "d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46",
        },
        type="PairClassification",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=(
            "2018-10-01",
            "2018-12-30",
        ),  # not found in the paper or data. This is just a rough guess based on the paper's publication date
        domains=["Programming", "Written"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from Sprint forum",
        bibtex_citation=r"""
@inproceedings{shah-etal-2018-adversarial,
  abstract = {We address the problem of detecting duplicate questions in forums, which is an important step towards automating the process of answering new questions. As finding and annotating such potential duplicates manually is very tedious and costly, automatic methods based on machine learning are a viable alternative. However, many forums do not have annotated data, i.e., questions labeled by experts as duplicates, and thus a promising solution is to use domain adaptation from another forum that has such annotations. Here we focus on adversarial domain adaptation, deriving important findings about when it performs well and what properties of the domains are important in this regard. Our experiments with StackExchange data show an average improvement of 5.6{\%} over the best baseline across multiple pairs of domains.},
  address = {Brussels, Belgium},
  author = {Shah, Darsh  and
Lei, Tao  and
Moschitti, Alessandro  and
Romeo, Salvatore  and
Nakov, Preslav},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1131},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {1056--1063},
  publisher = {Association for Computational Linguistics},
  title = {Adversarial Domain Adaptation for Duplicate Question Detection},
  url = {https://aclanthology.org/D18-1131},
  year = {2018},
}
""",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
