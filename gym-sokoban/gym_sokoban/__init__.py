import logging
import json
from importlib import resources
from gym.envs.registration import register

logger = logging.getLogger(__name__)

env_json = resources.files(__name__).joinpath("envs", "available_envs.json")

with open(env_json) as f:

    envs = json.load(f)

    for env in envs:
        register(
            id=env["id"],
            entry_point=env["entry_point"]
        )
