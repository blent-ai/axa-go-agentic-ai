"""Access to the models behind the LiteLLM gateway.

The LiteLLM gateway exposes an **OpenAI-compatible** API: we therefore use the
`ChatOpenAI` connector from `langchain-openai` and point it to the gateway URL
(`base_url`). No request goes to OpenAI when ``LITELLM_BASE_URL`` is set.

Environment variables (``.env`` file):

- ``LITELLM_BASE_URL``     : gateway URL, for example
  ``https://litellm.example.internal/v1``;
- ``LITELLM_API_KEY``      : virtual key issued by the gateway;
- ``LITELLM_MODEL``        : language model (tool calls, structured outputs);
- ``LITELLM_VISION_MODEL`` : vision model used to analyse photos. Defaults to
  ``LITELLM_MODEL`` (recent models are multimodal).

Without ``LITELLM_BASE_URL``, the project calls the OpenAI API directly with
``OPENAI_API_KEY``: the rest of the code is identical.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = os.environ.get("LITELLM_MODEL", "gpt-5.2")
DEFAULT_VISION_MODEL = os.environ.get("LITELLM_VISION_MODEL") or DEFAULT_MODEL


def get_chat_model(model: str | None = None, **kwargs) -> ChatOpenAI:
    """Build the language model used by the agents.

    Parameters
    ----------
    model : str, optional
        Name of the model to call. Defaults to the value of ``LITELLM_MODEL``.
    **kwargs
        Parameters forwarded to ``ChatOpenAI`` (``timeout``, ``max_retries``...).

    Returns
    -------
    ChatOpenAI
        The LangChain chat client.

    Raises
    ------
    RuntimeError
        If no API key is available in the environment.
    """
    base_url = os.environ.get("LITELLM_BASE_URL")
    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set LITELLM_API_KEY (LiteLLM gateway) or "
            "OPENAI_API_KEY in your .env file."
        )
    kwargs.setdefault("timeout", 120)
    kwargs.setdefault("max_retries", 2)
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=api_key,
        base_url=base_url or None,
        **kwargs,
    )


def get_vision_model(**kwargs) -> ChatOpenAI:
    """Build the vision model used to analyse photos.

    Returns
    -------
    ChatOpenAI
        The LangChain chat client, pointing to ``LITELLM_VISION_MODEL``.
    """
    return get_chat_model(DEFAULT_VISION_MODEL, **kwargs)
