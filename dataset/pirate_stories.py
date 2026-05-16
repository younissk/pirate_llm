import random

from datasets import load_from_disk

pirate_ds = load_from_disk("dataset/tiny_stories_pirate")

i = random.randint(0, len(pirate_ds["train"]) - 1)
print(pirate_ds["train"][i]["text"])
