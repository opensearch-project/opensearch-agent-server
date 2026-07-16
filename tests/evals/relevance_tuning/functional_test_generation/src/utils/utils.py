from itertools import chain


def flatten_lists(lists: list[list[any]]):
    return list(chain.from_iterable(lists))


def dict_to_key_value_lists(d: dict) -> list[list[str]]:
    return [[x, y] for x, y in d.items()]
