import locale
import json
from pydantic import BaseModel

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def format_value(value, float_decimals: int = 2) -> str | None:
    if isinstance(value, float):
        if int(value) == value:
            return str(int(value))
        else:
            return locale.format_string(f"%.{float_decimals}f", value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value)
    if value is None:
        return None
    return value


class FormattedModel(BaseModel):
    def print_representation(self, float_decimals: int = 2):
        dict = self.model_dump()
        return {key: format_value(value, float_decimals) for key, value in dict.items()}

