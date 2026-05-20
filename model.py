import os
from dotenv import load_dotenv
load_dotenv(override=True)

from strands.models.bedrock import BedrockModel

# Amazon Bedrock-hosted Claude Sonnet 4.5 (cross-region inference profile).
# AWS creds come from .env: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION.
MODEL = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    max_tokens=4096,
)
