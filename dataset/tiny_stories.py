from datasets import DatasetDict, load_dataset, load_from_disk

from dataset.piratize import piratize

if __name__ == "__main__":
    tiny_stories_ds = load_dataset("roneneldan/TinyStories")

    pirate_ds = DatasetDict({k: piratize(v) for k, v in tiny_stories_ds.items()})
    pirate_ds.save_to_disk("dataset/tiny_stories_pirate")
