from __future__ import annotations

from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "btk": ["bbc-Latn"],
    "bew": ["bew-Latn"],
    "bug": ["bug-Latn"],
    "jav": ["jav-Latn"],
    "mad": ["mad-Latn"],
    "mak": ["mak-Latn"],
    "min": ["min-Latn"],
    "mui": ["mui-Latn"],
    "rej": ["rej-Latn"],
    "sun": ["sun-Latn"],
}


class NusaParagraphEmotionClassification(MultilingualTask, AbsTaskClassification):
    metadata = TaskMetadata(
        name="NusaParagraphEmotionClassification",
        dataset={
            "path": "gentaiscool/nusaparagraph_emot",
            "revision": "c61e8c3ee47d2dce296e9601195916b54c21d575",
        },
        description="NusaParagraphEmotionClassification is a multi-class emotion classification on 10 Indonesian languages from the NusaParagraph dataset.",
        reference="https://github.com/IndoNLP/nusa-writes",
        category="s2s",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2021-08-01", "2022-07-01"),
        domains=["Non-fiction", "Fiction", "Written"],
        task_subtypes=["Emotion classification"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{cahyawijaya-etal-2023-nusawrites,
  address = {Nusa Dua, Bali},
  author = {Cahyawijaya, Samuel  and  Lovenia, Holy  and Koto, Fajri  and  Adhista, Dea  and  Dave, Emmanuel  and  Oktavianti, Sarah  and  Akbar, Salsabil  and  Lee, Jhonson  and  Shadieq, Nuur  and  Cenggoro, Tjeng Wawan  and  Linuwih, Hanung  and  Wilie, Bryan  and  Muridan, Galih  and  Winata, Genta  and  Moeljadi, David  and  Aji, Alham Fikri  and  Purwarianti, Ayu  and  Fung, Pascale},
  booktitle = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  editor = {Park, Jong C.  and  Arase, Yuki  and  Hu, Baotian  and  Lu, Wei  and  Wijaya, Derry  and  Purwarianti, Ayu  and  Krisnadhi, Adila Alfa},
  month = nov,
  pages = {921--945},
  publisher = {Association for Computational Linguistics},
  title = {NusaWrites: Constructing High-Quality Corpora for Underrepresented and Extremely Low-Resource Languages},
  url = {https://aclanthology.org/2023.ijcnlp-main.60},
  year = {2023},
}
""",
    )
