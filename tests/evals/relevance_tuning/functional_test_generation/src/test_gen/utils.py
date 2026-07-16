from collections.abc import Collection
import yaml
from pydantic import BaseModel


class Assertions:

    @staticmethod
    def get_llm_rubric_assertion_yaml(
        data: Collection[BaseModel],
        relevant_fields: Collection[str],
        field_key_to_description: dict[str, str],
        description_to_additional_values: dict[str, any],
        extra_notes: str = "",
    ):
        dicts = [x.model_dump(include=set(relevant_fields)) for x in data]
        dicts_mapped = [
            {field_key_to_description.get(key, key): value for key, value in x.items()}
            for x in dicts
        ]
        dicts_yaml_str = yaml.dump_all(dicts_mapped).replace('"', "'")
        additional_yaml_str = yaml.dump(description_to_additional_values).replace(
            '"', "'"
        )
        return f"""
        Below under the header '###SEQUENCE' a few expected results are listed in yaml format.
        Each root element in the yaml listing refers to a single result.
        The keys in the yaml format are a description of the meaning under which the
        corresponding value shall appear in the response text. If the key is a combination of
        words combined by some delimiter, try to derive the meaning from the sequence of
        words it is made of. If the key and the context under which the value is mentioned
        in the response do not match, fail the assertion.
        Under a second header '###FACTS' there might be additional key - value pairs
        to check for occurrence in the response text. Here again the key gives the context
        under which mentioning of the value is expected.
        If no key-value pairs occur in this section, then nothing is to be asserted here.
        Compare floating point values only up to the second decimal.
        Fail the test if the information provided below within both headers can not be checked
        against the response because some parts are missing. Statements such as some field
        or the response being too big and thus not being able to retrieve those also
        counts as fail. {extra_notes}

        ###SEQUENCE
        {dicts_yaml_str}

        ###FACTS
        {additional_yaml_str}
        """


