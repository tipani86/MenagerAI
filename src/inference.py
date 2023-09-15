# A collection of inference functions to be called from the main Streamlit app

import os
import json
import openai
import aiohttp
from settings import *

def convert_openai_messages_to_prompt(messages: list) -> str:
    # Convert the OpenAI standard messages format into a text prompt for models which don't support it
    prompt = "Below is a conversation between a human (user) and an AI (assistant). Please continue it by ONE ASSISTANT REPLY TEXT only, and do not add any extra comments.\n"
    for message in messages:
        match message["role"]:
            case "user":
                prompt += f"User: {message['content']}\n"
            case "assistant":
                prompt += f"Assistant: {message['content']}\n"
            # For now, we ignore other possible roles, but may revisit the logic later
    prompt += "Assistant: "
    return prompt


async def call_openai(
    messages: list,
    model_settings: dict,
) -> str | None:
    try:
        # Just make sure the environment variables are set, OpenAI API can get them automatically
        os.getenv("OPENAI_API_KEY")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")

    stream = model_settings.get("stream", False)
    if stream:
        async for chunk in await openai.ChatCompletion.acreate(
            model=model_settings["model_name"],
            engine=model_settings["model_engine"],
            messages=messages,
            max_tokens=model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
            stream=stream,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content", None)
            if content is not None:
                yield content
    else:
        resp = await openai.ChatCompletion.acreate(
            model=model_settings["model_name"],
            engine=model_settings["model_engine"],
            messages=messages,
            max_tokens=model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
        )
        yield resp["choices"][0]["message"]["content"].strip()


async def call_deepinfra(
    messages: list,
    model_settings: dict,
):
    try:
        api_key = os.getenv("DEEPINFRA_API_KEY")
        api_base = os.getenv("DEEPINFRA_API_BASE").rstrip("/")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")
    
    method = "POST"
    url=f"{api_base}/{model_settings['model_name']}"
    stream = model_settings.get("stream", False)
    
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = convert_openai_messages_to_prompt(messages)

    data = {
        "input": prompt,
        "max_new_tokens": model_settings.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
        "stream": stream,
    }

    async for resp in _call_api(
        method=method,
        url=url,
        headers=headers,
        data=data,
        stream=stream,
    ):
        if stream:
            if resp.startswith("data: "):
                resp = json.loads(resp.split("data: ", 1)[1])
                content = resp.get("token", {}).get("text", None)
                if content == "</s>":   # Special eos token
                    break
                if content is not None:
                    yield content
        else:
            if isinstance(resp, str) and resp.startswith(prompt):
                resp = resp.split(prompt, 1)[1]
            yield resp


async def _call_api(
    method: str,
    url: str,
    headers: dict = {},
    params: dict = {},
    data: dict = {},
    stream: bool = False,
):
    async with aiohttp.ClientSession() as session:
        async with session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=data,
        ) as resp:
            if stream:
                async for line in resp.content:
                    chunk = line.decode("utf-8").strip()
                    yield chunk
            else:
                res_json = await resp.json()
                yield res_json["results"][0]["generated_text"].strip()