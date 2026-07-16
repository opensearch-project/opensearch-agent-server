import json
import os
import uuid
from collections.abc import Iterator
import requests
from dotenv import load_dotenv
from utils.logging_helpers import get_logger
from server.logging_config import configure_logging, get_logging_config_from_env

logger = get_logger(__name__)
script_path = "/".join(os.path.realpath(__file__).split("/")[:-1])

# Load environment variables from .env file
load_dotenv(dotenv_path=f"{script_path}/../../../.env")
logger.info("Loaded env vars from global .env file")

# logging
use_json, log_level = get_logging_config_from_env()
configure_logging(use_json=use_json, log_level=log_level, force=True)

AG_UI_HOST = os.environ["AG_UI_HOST"]
AG_UI_PORT = os.environ["AG_UI_PORT"]

OS_USERNAME = os.environ.get("OPENSEARCH_USERNAME", None)
OS_PASSWORD = os.environ.get("OPENSEARCH_PASSWORD", None)


def consume_response_text(iterator: Iterator) -> str:
    message = ""
    for line in iterator:
        if line:
            decoded_line = line.decode("utf-8").strip().removeprefix("data: ")
            try:
                json_content = json.loads(decoded_line)
                if json_content.get("type", "unknown") == "TEXT_MESSAGE_CONTENT":
                    message += json_content.get("delta", "")
            except Exception as ex:
                logger.warning(f"Decoding message: {decoded_line} failed w exception '{str(ex)}'.")

    return message


async def call_api(prompt: str, options: dict, context: dict) -> dict:
    from ag_ui.core import RunAgentInput

    """
    Assumes running ag_ui_server
    """
    # pick those up if configured in test cases, to reflect different tests on different threads
    # we add uuid here if threadId / runId are not set to avoid previous redteam attacks staying in context
    thread_id = (
        context.get("vars", {})
        .get("metadata", {})
        .get("threadId", f"thread-{uuid.uuid4()}")
    )
    run_id = (
        context.get("vars", {}).get("metadata", {}).get("runId", f"run-{uuid.uuid4()}")
    )
    url = f"http://{AG_UI_HOST}:{AG_UI_PORT}/runs"
    input_body = RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        messages=[{"id": "msg-1", "role": "user", "content": prompt}],
        state={},
        tools=[],
        context=[],
        forwarded_props={"page_context": "searchRelevance"},
    )
    data = input_body.model_dump()
    response = requests.post(
        url,
        json=data,
        stream=True,
        auth=(OS_USERNAME, OS_PASSWORD) if (OS_USERNAME and OS_PASSWORD) else None,
        headers={"Host": AG_UI_HOST, "Connection": "keep-alive"}
    )
    status_code = response.status_code

    message = consume_response_text(response.iter_lines())

    if status_code >= 400:
        return {"error": message}
    return {"output": message}
