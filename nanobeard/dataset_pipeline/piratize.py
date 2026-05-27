from multiprocessing import Pool, cpu_count

from arrr import translate
from datasets import Dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

load_dotenv()
console = Console()


def _translate_one(text: str) -> str:
    return translate(text)


def piratize(dataset: Dataset, split_name: str = "") -> Dataset:
    texts = dataset["text"]
    total = len(texts)
    print(f"total {total:,}")
    out = []

    with (
        Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
        Pool(cpu_count() - 1) as pool,
    ):
        task = progress.add_task(f"⚓ {split_name}", total=total)
        for pirate in pool.imap(_translate_one, texts, chunksize=64):
            out.append(pirate)
            progress.advance(task)

    return Dataset.from_dict({"text": out})
