
import re
import uuid
import time
import openai
from typing import List

def row_create(
    model: str,
    custom_id: str,
    messages: List[dict],
    n: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    logprobs: bool = False,
    top_logprobs: int = 5,
    top_p: float = 0.95,
    stop: str = None,
) -> dict:
    """
    Create a row for the batch file.
    """
    params =  {
        "method": "POST",
        "url": "/v1/chat/completions",
        "custom_id": custom_id,
        "body": {
            "n": n,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            # "logprobs": logprobs,
            "messages": messages,
            "stop": stop
        }
    }
    if logprobs:
        params["body"]["top_logprobs"] = top_logprobs
    return params

def prompt_to_message(
    prompt: str,
    system_message: str = ""
) -> dict:
    """
    Convert the prompt to a message list.
    """
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    return messages

def batch_create(
    client: openai.OpenAI,
    input_file: str,
) -> str:
    """
    Create a openai batch task.
    """
    batch_input_file = client.files.create(
            file=open(input_file, "rb"),
            purpose="batch"
        )
    batch_input_file_id = batch_input_file.id
    print("Batch input file id: ", batch_input_file_id)
    task_id = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "nightly eval job"
        }
    ).id
    return task_id

def run_batch_instance(
    row: dict,
    client: openai.OpenAI
) -> dict:
    """
    Run a batch instance using instant API.
    """
    result = {
        "id": "custom-" + str(uuid.uuid3(uuid.NAMESPACE_DNS, "custom-" + row["custom_id"] + str(time.time()))),
        "custom_id": row["custom_id"],
        "response": {
            "status": 200,
            "request_id":  "custom-batch-" + str(uuid.uuid3(uuid.NAMESPACE_DNS, "custom-batch-" + row["custom_id"] + str(time.time()))),
            "body": None
        },
        "error": None
    }
    try:
        if "chat" in row["url"]:
            result["response"]["body"] = client.chat.completions.create(
                **row["body"]
            ).to_dict()
        else:
            result["response"]["body"] = client.completions.create(
                **row["body"]
            ).to_dict()
    except Exception as e:
        try:
            error = eval(re.findall(r"\{.+\}", e.args[0])[0])
        except:
            error = {"code": -1}
        result["response"]["status"] = error['code']
        result["error"] = e
    return result

