from collections.abc import Collection

from pydantic import BaseModel, ConfigDict

from src.utils.utils import flatten_lists

"""
Module for representing assertion cases and representing combinations of assertions
and different test cases correctly in csv format.
"""


class Assertion(BaseModel):
    def get_csv_assertion_text(self):
        pass


class LLMRubricAssertion(Assertion):
    eval_prompt: str

    def get_csv_assertion_text(self):
        return f"llm-rubric: {self.eval_prompt}"


class CaseInsensitiveContainsAssertion(Assertion):
    contains_all: bool = True
    contains_texts: list[str] | tuple[str, ...]

    def get_csv_assertion_text(self):
        return f"icontains-{'any' if not self.contains_all else 'all'}: {','.join(self.contains_texts)}"


class TestCase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    assertions: list[Assertion] | tuple[Assertion]
    metadata: dict[str, str] | None = None

    def get_csv_test_row(self) -> str:
        return (
            f'"{self.prompt}"'
            + ","
            + ",".join(
                ['"' + x.get_csv_assertion_text() + '"' for x in self.assertions]
            )
        )


class TestSuite(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    cases: Collection[TestCase]

    def get_ordered_metadata_fields(self) -> Collection[str]:
        return sorted(
            list(
                set(
                    flatten_lists(
                        [
                            list(x.metadata.keys())
                            for x in self.cases
                            if x.metadata is not None
                        ]
                    )
                )
            )
        )

    def get_csv_header(self) -> str:
        header = ",".join(
            ["prompt"]
            + [f"__expected{x}" for x in range(1, self.max_num_assertions() + 1)]
        )
        meta_data_keys = self.get_ordered_metadata_fields()
        header = ",".join([header] + [f"__metadata:{x}" for x in meta_data_keys])
        return header

    def get_csv_rows(self) -> list[str] | tuple[str]:
        rows = []
        for tcase in self.cases:
            num_empty_asserts = self.max_num_assertions() - len(tcase.assertions)
            rows.append(
                ",".join(
                    [tcase.get_csv_test_row()] + ["" for _ in range(num_empty_asserts)]
                )
            )
            # now also append the metadata where set. If not available for a case,
            # leave out
            meta_data_fields = self.get_ordered_metadata_fields()
            if len(meta_data_fields) > 0:
                meta_data_suffix = ",".join(
                    [
                        tcase.metadata.get(x, "") if tcase.metadata else ""
                        for x in self.get_ordered_metadata_fields()
                    ]
                )
                rows[-1] = ",".join([rows[-1], meta_data_suffix])
        return rows

    def write(self, path: str):
        with open(path, "w") as f:
            f.write("\n".join([self.get_csv_header()] + self.get_csv_rows()))

    def max_num_assertions(self) -> int:
        """
        Max nr of assertions. We will need that many assertion columns in csv where each column is named
        __expected[index + 1], e.g _expected1, _expected2, ..., _expectedN
        """
        return max([len(x.assertions) for x in self.cases])
