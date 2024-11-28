# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.
"""
import random
import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]
# fmt: on

# ArabicMMLU
# fmt: off
ARABIC_MMLU_SUBSETS = [
    "Islamic Studies", "Islamic Studies (Middle School)", "Islamic Studies (Primary School)", "Islamic Studies (High School)", "Driving Test",
    "Natural Science (Middle School)", "Natural Science (Primary School)", "History (Middle School)", "History (Primary School)", "History (High School)", "General Knowledge",
    "General Knowledge (Middle School)", "General Knowledge (Primary School)", "Law (Professional)", "Physics (High School)", "Social Science (Middle School)",
    "Social Science (Primary School)", "Management (University)", "Arabic Language (Middle School)", "Arabic Language (Primary School)", "Arabic Language (High School)", "Political Science (University)",
    "Philosophy (High School)", "Accounting (University)", "Computer Science (Middle School)", "Computer Science (Primary School)", "Computer Science (High School)", "Computer Science (University)",
    "Geography (Middle School)", "Geography (Primary School)", "Geography (High School)", "Math (Primary School)", "Biology (High School)", "Economics (Middle School)",
    "Economics (High School)", "Economics (University)", "Arabic Language (General)", "Arabic Language (Grammar)", "Civics (Middle School)", "Civics (High School)"
]
# fmt: on


def mmlu_darija(line, task_name: str = None):
    topic = line["subject"]
    instruction = f"الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول {topic.replace('_', ' ')}. \n\n"
    choices = line["choices"]
    # Answers are provided with roman letters - we look for the correct index in LETTER_INDICES,
    # it will then be applied to arabic letters
    gold_ix = line["answer"]

    query = f"{instruction}{line['question']}\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES_AR[:4], choices)])
    query += "الإجابة:"

    return Doc(
        task_name=task_name,
        query=query,
        choices=LETTER_INDICES_AR[:4],
        gold_index=gold_ix,
        instruction=instruction,
    )


class CustomDarijaMMLUTask(LightevalTaskConfig):
    def __init__(
        self,
        name,
        hf_subset,
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=mmlu_darija,
            hf_repo="MBZUAI-Paris/DarijaMMLU",
            metric=[Metrics.loglikelihood_acc_norm],
            hf_avail_splits=["test", "dev"],
            evaluation_splits=["test"],
            few_shots_split="dev",
            few_shots_select="sequential",
            suite=["community"],
            generation_size=-1,
            stop_sequence=None,
            output_regex=None,
            frozen=False,
            trust_dataset=True,
            version=0,
        )


DARIJA_MMLU_TASKS = [
    CustomDarijaMMLUTask(name=f"darija_mmlu:{subset}", hf_subset=subset) for subset in ARABIC_MMLU_SUBSETS
]

TASKS_TABLE = (
    DARIJA_MMLU_TASKS
)

if __name__ == "__main__":
    print(t.name for t in TASKS_TABLE)
    print(len(TASKS_TABLE))
