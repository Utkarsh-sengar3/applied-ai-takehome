import os
from pydantic import BaseModel, Field

class JudgeSettings(BaseModel):
    # Model Configurations (Decoupled Judge and Generator)
    judge_model: str = Field(default=os.getenv("JUDGE_MODEL", "meta-llama/llama-3.3-70b-instruct"))
    generator_model: str = Field(default=os.getenv("GENERATOR_MODEL", "mistralai/mistral-7b-instruct"))
    
    # API Keys
    openrouter_api_key: str = Field(default=os.getenv("OPENROUTER_API_KEY", ""))
    judge_api_key: str = Field(default=os.getenv("JUDGE_API_KEY", os.getenv("OPENROUTER_API_KEY", "")))
    generator_api_key: str = Field(default=os.getenv("GENERATOR_API_KEY", os.getenv("OPENROUTER_API_KEY", "")))

    # Judge Execution Parameters
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    pass_threshold: float = Field(default=3.5)

    # Bias Mitigations
    enable_position_swap: bool = Field(default=True)
    enable_length_penalty: bool = Field(default=True)
    max_verbosity_ratio: float = Field(default=2.5)

    # Logging
    audit_log_path: str = Field(default=os.getenv("AUDIT_LOG_PATH", "./logs/judge_audit.jsonl"))

settings = JudgeSettings()
