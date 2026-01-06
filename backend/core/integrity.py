import hashlib
import json

def generate_job_signature(job_config: dict) -> str:
    config_str = json.dumps(job_config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

def verify_job_signature(job_config: dict, signature: str) -> bool:
    actual_signature = generate_job_signature(job_config)
    return actual_signature == signature

