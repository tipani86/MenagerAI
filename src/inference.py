# A collection of inference functions to be called from the main Streamlit app

import os
import openai
import aiohttp
from settings import *

async def call_openai(
    model: dict,
    messages: list,
    stream: bool = False,
) -> str | None:
    try:
        # Just make sure the environment variables are set, OpenAI API can get them automatically
        os.getenv("OPENAI_API_KEY")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")
    
    if stream:
        async for chunk in await openai.ChatCompletion.acreate(
            model=model["model_name"],
            engine=model["model_engine"],
            messages=messages,
            max_tokens=model.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
            stream=stream,
        ):
            content = chunk["choices"][0].get("delta", {}).get("content", None)
            if content is not None:
                yield content
    else:
        resp = await openai.ChatCompletion.acreate(
            model=model["model_name"],
            engine=model["model_engine"],
            messages=messages,
            max_tokens=model.get("max_reply_tokens", LLM_MAX_REPLY_TOKENS),
        )
        yield resp["choices"][0]["message"]["content"].strip()
    

async def call_deepinfra(
    method: str,
    model: dict,
    messages: list,
    stream: bool = False,
):
    try:
        api_key = os.getenv("DEEPINFRA_API_KEY")
        api_base = os.getenv("DEEPINFRA_API_BASE")
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable: {e}")
    
    


async def _call_api(
    method: str,
    url: str,
    headers: dict = {},
    params: dict = {},
    data: dict = {},
    stream: bool = False,
):
    pass