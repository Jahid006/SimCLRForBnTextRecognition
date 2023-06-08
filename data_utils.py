from collections import namedtuple, defaultdict
import random
import json


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
        return f"{self.text}: " + "\n".join(
            [f"{bucket}: {len(data)}" for bucket, data in self.buckets.items()]
        )


class DataSourceController:
    def __init__(
        self,
        data: dict = None,
        base_dir: str = "",
        id: str = "",
        n: int = None,
        filter_: callable = lambda x: True,
        transform: callable = lambda x: x,
    ):
        self.data = []
        self.format = namedtuple("data", ["id", "path", "label"])
        self.ids = defaultdict(int)
        self.filter = filter_
        self.transform = transform

        if data and base_dir and id:
            self.add_data(data, base_dir, id, n)

    def add_data(
        self, data: dict = {}, base_dir: str = "", id: str = "na", take_n=None
    ):
        n = take_n

        if isinstance(data, str):
            try:
                data = self.read_json(data)
            except Exception as e:
                print("Data not vaild")

        _data = {
            k: self.transform(v)
            for k, v in data.items()
            if self.filter(self.format(id, f"{base_dir}/{k}", data[k]))
        }

        _keys = list(_data.keys())

        if n:
            n = min(n, len(_data))
            _keys = random.sample(_keys, n)

        _data = [self.format(id, f"{base_dir}/{k}", _data[k]) for k in _keys]

        self.data.extend(_data)
        self.ids[id] += len(_data)

        print(f"Out of {len(data)} {id}, {len(_data)} are kept after filtering.")
        print(f"Total data {len(self.data)}")

    def modifiy_filter(self, function: callable, update_data=False):
        self.filter = function

        if update_data:
            l = len(self.data)
            self.data = [d for d in self.data if self.filter(d)]
            self.ids = defaultdict(int)

            for d in self.data:
                self.ids[d.id] += 1

            print(f"Out of {l}, {len(self.data)} are kept after filtering.")

    def shuffle(self, random_state=42):
        random.shuffle(self.data)

    def read_json(self, path, encoding="utf-8"):
        return json.load(open(path, "r", encoding=encoding))

    @property
    def unique_source(self):
        return self.ids.keys()

    def __repr__(self) -> str:
        return (
            f"Total Data: {len(self.data)}"
            + "\n"
            + "\n\t".join([f"{k}: {v}" for k, v in self.ids.items()])
        )


