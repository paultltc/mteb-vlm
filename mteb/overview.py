"""This script contains functions that are used to get an overview of the MTEB benchmark."""

from __future__ import annotations

import difflib
import logging
from collections import Counter, defaultdict

import pandas as pd

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.TaskMetadata import TASK_CATEGORY, TASK_DOMAIN, TASK_TYPE
from mteb.custom_validators import MODALITIES
from mteb.languages import (
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
    path_to_lang_codes,
    path_to_lang_scripts,
)
from mteb.tasks import *  # import all tasks

logger = logging.getLogger(__name__)


# Create task registry


def create_task_list() -> list[type[AbsTask]]:
    tasks_categories_cls = list(AbsTask.__subclasses__())
    tasks = [
        cls
        for cat_cls in tasks_categories_cls
        for cls in cat_cls.__subclasses__()
        if cat_cls.__name__.startswith("AbsTask")
    ]
    return tasks


def create_name_to_task_mapping() -> dict[str, type[AbsTask]]:
    tasks = create_task_list()
    return {cls.metadata.name: cls for cls in tasks}


def create_similar_tasks() -> dict[str, list[str]]:
    """Create a dictionary of similar tasks.

    Returns:
        Dict with key is parent task and value is list of similar tasks.
    """
    tasks = create_task_list()
    similar_tasks = defaultdict(list)
    for task in tasks:
        if task.metadata.adapted_from:
            for similar_task in task.metadata.adapted_from:
                similar_tasks[similar_task].append(task.metadata.name)
    return similar_tasks


TASKS_REGISTRY = create_name_to_task_mapping()
SIMILAR_TASKS = create_similar_tasks()


def check_is_valid_script(script: str) -> None:
    if script not in ISO_TO_SCRIPT:
        raise ValueError(
            f"Invalid script code: {script}, you can find valid ISO 15924 codes in {path_to_lang_scripts}"
        )


def check_is_valid_language(lang: str) -> None:
    if lang not in ISO_TO_LANGUAGE:
        raise ValueError(
            f"Invalid language code: {lang}, you can find valid ISO 639-3 codes in {path_to_lang_codes}"
        )


def filter_superseded_datasets(tasks: list[AbsTask]) -> list[AbsTask]:
    return [t for t in tasks if t.superseded_by is None]


def filter_tasks_by_languages(
    tasks: list[AbsTask], languages: list[str]
) -> list[AbsTask]:
    [check_is_valid_language(lang) for lang in languages]
    langs_to_keep = set(languages)
    return [t for t in tasks if langs_to_keep.intersection(t.metadata.languages)]


def filter_tasks_by_script(tasks: list[AbsTask], script: list[str]) -> list[AbsTask]:
    [check_is_valid_script(s) for s in script]
    script_to_keep = set(script)
    return [t for t in tasks if script_to_keep.intersection(t.metadata.scripts)]


def filter_tasks_by_domains(
    tasks: list[AbsTask], domains: list[TASK_DOMAIN]
) -> list[AbsTask]:
    domains_to_keep = set(domains)

    def _convert_to_set(domain: list[TASK_DOMAIN] | None) -> set:
        return set(domain) if domain is not None else set()

    return [
        t
        for t in tasks
        if domains_to_keep.intersection(_convert_to_set(t.metadata.domains))
    ]


def filter_tasks_by_task_types(
    tasks: list[AbsTask], task_types: list[TASK_TYPE]
) -> list[AbsTask]:
    _task_types = set(task_types)
    return [t for t in tasks if t.metadata.type in _task_types]


def filter_task_by_categories(
    tasks: list[AbsTask], categories: list[TASK_CATEGORY]
) -> list[AbsTask]:
    _categories = set(categories)
    return [t for t in tasks if t.metadata.category in _categories]


def filter_tasks_by_modalities(
    tasks: list[AbsTask],
    modalities: list[MODALITIES],
    exclude_modality_filter: bool = False,
) -> list[AbsTask]:
    _modalities = set(modalities)
    if exclude_modality_filter:
        return [t for t in tasks if set(t.modalities) == _modalities]
    else:
        return [t for t in tasks if _modalities.intersection(t.modalities)]


def filter_aggregate_tasks(tasks: list[AbsTask]) -> list[AbsTask]:
    """Returns input tasks that are *not* aggregate.

    Args:
        tasks: A list of tasks to filter.
    """
    return [t for t in tasks if not t.is_aggregate]


class MTEBTasks(tuple):
    def __repr__(self) -> str:
        return "MTEBTasks" + super().__repr__()

    @staticmethod
    def _extract_property_from_task(task, property):
        if hasattr(task.metadata, property):
            return getattr(task.metadata, property)
        elif hasattr(task, property):
            return getattr(task, property)
        elif property in task.metadata_dict:
            return task.metadata_dict[property]
        else:
            raise KeyError("Property neither in Task attribute or in task metadata.")

    @property
    def languages(self) -> set:
        """Return all languages from tasks"""
        langs = set()
        for task in self:
            for lg in task.languages:
                langs.add(lg)
        return langs

    def count_languages(self) -> dict:
        """Summarize count of all languages from tasks"""
        langs = []
        for task in self:
            langs.extend(task.languages)
        return Counter(langs)

    def to_markdown(
        self, properties: list[str] = ["type", "license", "languages", "modalities"]
    ) -> str:
        """Generate markdown table with tasks summary

        Args:
            properties: list of metadata to summarize from a Task class.

        Returns:
            string with a markdown table.
        """
        markdown_table = "| Task" + "".join([f"| {p} " for p in properties]) + "|\n"
        _head_sep = "| ---" * len(properties) + " |\n"
        markdown_table += _head_sep
        for task in self:
            markdown_table += f"| {task.metadata.name}"
            markdown_table += "".join(
                [f"| {self._extract_property_from_task(task, p)}" for p in properties]
            )
            markdown_table += " |\n"
        return markdown_table

    def to_dataframe(
        self,
        properties: list[str] = [
            "name",
            "type",
            "languages",
            "domains",
            "license",
            "modalities",
        ],
    ) -> pd.DataFrame:
        """Generate pandas DataFrame with tasks summary

        Args:
            properties: list of metadata to summarize from a Task class.

        Returns:
            pandas DataFrame.
        """
        data = []
        for task in self:
            data.append(
                {p: self._extract_property_from_task(task, p) for p in properties}
            )
        return pd.DataFrame(data)

    def to_latex(
        self,
        properties: list[str] = [
            "name",
            "type",
            "languages",
            "domains",
            "license",
            "modalities",
        ],
        group_indices: list[str] | None = ["type", "name"],
        include_citation_in_name: bool = True,
        limit_n_entries: int | None = 3,
    ) -> str:
        """Generate a LaTeX table of the tasks.

        Args:
            properties: list of metadata to summarize from a Task class.
            group_indices: list of properties to group the table by.
            include_citation_in_name: Whether to include the citation in the name.
            limit_n_entries: Limit the number of entries for cell values, e.g. number of languages and domains. Will use "..." to indicate that
                there are more entries.
        """
        if include_citation_in_name and "name" in properties:
            properties += ["intext_citation"]
            df = self.to_dataframe(properties)
            df["name"] = df["name"] + " " + df["intext_citation"]
            df = df.drop(columns=["intext_citation"])
        else:
            df = self.to_dataframe(properties)

        if limit_n_entries and df.shape[0]:  # ensure that there are entries
            for col in df.columns:
                # check if content is a list or set
                if isinstance(df[col].iloc[0], (list, set)):
                    _col = []
                    for val in df[col]:
                        if val is not None and len(val) > limit_n_entries:
                            ending = "]" if isinstance(val, list) else "}"
                            str_col = str(val[:limit_n_entries])[:-1] + ", ..." + ending
                        else:
                            str_col = str(val)

                        # escape } and { characters
                        str_col = str_col.replace("{", "\\{").replace("}", "\\}")
                        _col.append(str_col)
                    df[col] = _col

        if group_indices:
            df = df.set_index(group_indices)

        return df.to_latex()


def get_tasks(
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TASK_DOMAIN] | None = None,
    task_types: list[TASK_TYPE] | None = None,
    categories: list[TASK_CATEGORY] | None = None,
    tasks: list[str] | None = None,
    exclude_superseded: bool = True,
    eval_splits: list[str] | None = None,
    exclusive_language_filter: bool = False,
    modalities: list[MODALITIES] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_aggregate: bool = False,
) -> MTEBTasks:
    """Get a list of tasks based on the specified filters.

    Args:
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn". For multilingual tasks this will also remove languages that are not in the specified list.
        script: A list of script codes (ISO 15924 codes). If None, all scripts are included. For multilingual tasks this will also remove scripts
            that are not in the specified list.
        domains: A list of task domains.
        task_types: A string specifying the type of task. If None, all tasks are included.
        categories: A list of task categories these include "s2s" (sentence to sentence), "s2p" (sentence to paragraph) and "p2p" (paragraph to
            paragraph).
        tasks: A list of task names to include. If None, all tasks which pass the filters are included.
        exclude_superseded: A boolean flag to exclude datasets which are superseded by another.
        eval_splits: A list of evaluation splits to include. If None, all splits are included.
        exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
            exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
            specified will be kept.
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.
        exclude_aggregate: If True, exclude aggregate tasks. If False, both aggregate and non-aggregate tasks are returned.

    Returns:
        A list of all initialized tasks objects which pass all of the filters (AND operation).

    Examples:
        >>> get_tasks(languages=["eng", "deu"], script=["Latn"], domains=["Legal"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Classification"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Clustering"], exclude_superseded=False)
        >>> get_tasks(languages=["eng"], tasks=["WikipediaRetrievalMultilingual"], eval_splits=["test"])
        >>> get_tasks(tasks=["STS22"], languages=["eng"], exclusive_language_filter=True) # don't include multilingual subsets containing English
    """
    if tasks:
        _tasks = [
            get_task(
                task,
                languages,
                script,
                eval_splits=eval_splits,
                exclusive_language_filter=exclusive_language_filter,
                modalities=modalities,
                exclusive_modality_filter=exclusive_modality_filter,
            )
            for task in tasks
        ]
        return MTEBTasks(_tasks)

    _tasks = [
        cls().filter_languages(languages, script).filter_eval_splits(eval_splits)
        for cls in create_task_list()
    ]

    if languages:
        _tasks = filter_tasks_by_languages(_tasks, languages)
    if script:
        _tasks = filter_tasks_by_script(_tasks, script)
    if domains:
        _tasks = filter_tasks_by_domains(_tasks, domains)
    if task_types:
        _tasks = filter_tasks_by_task_types(_tasks, task_types)
    if categories:
        logger.warning(
            "`s2p`, `p2p`, and `s2s` will be removed and replaced by `t2t` in v2.0.0."
        )
        _tasks = filter_task_by_categories(_tasks, categories)
    if exclude_superseded:
        _tasks = filter_superseded_datasets(_tasks)
    if modalities:
        _tasks = filter_tasks_by_modalities(
            _tasks, modalities, exclusive_modality_filter
        )
    if exclude_aggregate:
        _tasks = filter_aggregate_tasks(_tasks)

    return MTEBTasks(_tasks)


_TASK_RENAMES = {"PersianTextTone": "SynPerTextToneClassification"}


def get_task(
    task_name: str,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    eval_splits: list[str] | None = None,
    hf_subsets: list[str] | None = None,
    exclusive_language_filter: bool = False,
    modalities: list[MODALITIES] | None = None,
    exclusive_modality_filter: bool = False,
) -> AbsTask:
    """Get a task by name.

    Args:
        task_name: The name of the task to fetch.
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn". For multilingual tasks this will also remove languages that are not in the specified list.
        script: A list of script codes (ISO 15924 codes). If None, all scripts are included. For multilingual tasks this will also remove scripts
        eval_splits: A list of evaluation splits to include. If None, all splits are included.
        hf_subsets: A list of Huggingface subsets to evaluate on.
        exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
            exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
            specified will be kept.
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.

    Returns:
        An initialized task object.

    Examples:
        >>> get_task("BornholmBitextMining")
    """
    if task_name in _TASK_RENAMES:
        _task_name = _TASK_RENAMES[task_name]
        logger.warning(
            f"The task with the given name '{task_name}' has been renamed to '{_task_name}'. To prevent this warning use the new name."
        )

    if task_name not in TASKS_REGISTRY:
        close_matches = difflib.get_close_matches(task_name, TASKS_REGISTRY.keys())
        if close_matches:
            suggestion = (
                f"KeyError: '{task_name}' not found. Did you mean: {close_matches[0]}?"
            )
        else:
            suggestion = (
                f"KeyError: '{task_name}' not found and no similar keys were found."
            )
        raise KeyError(suggestion)
    task = TASKS_REGISTRY[task_name]()
    if eval_splits:
        task.filter_eval_splits(eval_splits=eval_splits)
    if modalities:
        task.filter_modalities(modalities, exclusive_modality_filter)
    return task.filter_languages(
        languages,
        script,
        hf_subsets=hf_subsets,
        exclusive_language_filter=exclusive_language_filter,
    )
