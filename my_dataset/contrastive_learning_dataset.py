import random
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class Example:
    def __init__(self, text, sources):
        self.text = text
        self.sources = sources
        random.shuffle(self.sources)

        self.buckets = defaultdict(list)
        for source in self.sources:
            self.buckets[source.id].append(source.path)

    def __call__(self, k=2):
        views = random.choices(self.sources, k=k)
        return [view.path for view in views]

    def __repr__(self) -> str:
        return f"{self.text}: " + "\n".join([
            f"{bucket}: {len(data)}"
            for bucket, data in self.buckets.items()
        ])


class ContrastiveLearningDataset(Dataset):
    def __init__(
        self,
        data,
        img_height=32,
        img_width=128,
        noiseAugment=None,
    ):
        self.data = data
        self.img_height = img_height
        self.img_width = img_width
        self.noiseAugment = noiseAugment
        self.probability = self.noiseAugment.probability
        self.canvas = Image.new("L", (img_width, img_height), color="white")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            example = self.data[index]
            views = [example.path] * 2
            probability = 1 if views[0] == views[1] else self.probability
            return [self.get_image(v, probability) for v in views]
        except Exception as e:
            print(e, example.path)
            return self[index + 1]

    def get_image(self, path, probability=None):
        image = Image.open(path).convert("L")

        if self.noiseAugment:
            image = np.array(image)
            image = self.noiseAugment(image, probability=probability)
            image = Image.fromarray(image.astype(np.uint8))

        image = self.paste_image(image.copy(), self.canvas.copy())
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        return image

    def paste_image(self, image: Image, canvas: Image):
        cw, ch = canvas.size
        image.thumbnail((cw, ch), Image.ANTIALIAS)
        w, h = image.size

        if w < cw:
            canvas.paste(image, (int((cw - w) * random.random()), 0))
        elif h < ch:
            canvas.paste(image, (0, int((ch - h) * random.random())))
        elif w == cw and h == ch:
            canvas = image
        else:
            image.resize((cw, ch))
            canvas = image
        return canvas

    # def __getitem__(self, index):
    #     _data = self.data[index]
    #     views = (
    #         [random.choice(self.data_bucket[_data]).path]
    #         + [random.choice(self.data_bucket[_data]).path]
    #     )
    #     probability = 1 if views[0] == views[1] else self.probability
    #     return [self.get_image(v, probability) for v in views]
