import os
from dotenv import load_dotenv
load_dotenv(override=True)

from strands.models.anthropic import AnthropicModel

MODEL = AnthropicModel(
    client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
    model_id="claude-sonnet-4-5-20250929",
    max_tokens=2048,
)
