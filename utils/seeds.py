import json
import random
import numpy as np
import os

SEED_LOG_FILE = "seed_log.json"

def set_seed(seed=42, context="general"):
    """Set random seed for reproducibility and log it."""
    random.seed(seed)
    np.random.seed(seed)
    
    log_seed(seed, context)

def log_seed(seed, context):
    """Log the used seed."""
    log_data = {}
    if os.path.exists(SEED_LOG_FILE):
        try:
            with open(SEED_LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    log_data[context] = seed
    
    with open(SEED_LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)
        
def get_seed(context="general", default=42):
    """Retrieve a seed for a context, or generate and log a new one."""
    if os.path.exists(SEED_LOG_FILE):
        try:
            with open(SEED_LOG_FILE, 'r') as f:
                log_data = json.load(f)
                if context in log_data:
                    return log_data[context]
        except json.JSONDecodeError:
            pass
            
    set_seed(default, context)
    return default
