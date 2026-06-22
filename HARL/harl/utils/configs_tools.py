"""Tools for loading and updating configs."""
import time
import os
import json
import re
import yaml
from uu import Error

try:
    import torch
except ImportError:
    torch = None


class DummySummaryWriter:
    """Fallback writer used when tensorboard logging cannot be initialized."""

    def __init__(self):
        self.scalars = {}

    @staticmethod
    def _to_float(value):
        if torch is not None and torch.is_tensor(value):
            value = value.detach().cpu().item()
        return float(value)

    def add_scalars(self, *args, **kwargs):
        if len(args) >= 2:
            main_tag = args[0]
            tag_scalar_dict = args[1]
            global_step = args[2] if len(args) >= 3 else kwargs.get("global_step", 0)
            walltime = time.time()
            for tag, value in tag_scalar_dict.items():
                key = f"{main_tag}/{tag}" if main_tag != tag else str(main_tag)
                self.scalars.setdefault(key, []).append(
                    [walltime, int(global_step), self._to_float(value)]
                )
        return None

    def add_scalar(self, *args, **kwargs):
        if len(args) >= 2:
            tag = args[0]
            scalar_value = args[1]
            global_step = args[2] if len(args) >= 3 else kwargs.get("global_step", 0)
            walltime = time.time()
            self.scalars.setdefault(str(tag), []).append(
                [walltime, int(global_step), self._to_float(scalar_value)]
            )
        return None

    def export_scalars_to_json(self, path, *args, **kwargs):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.scalars, file, ensure_ascii=False)
        return None

    def close(self):
        return None


def get_defaults_yaml_args(algo, env):
    """Load config file for user-specified algo and env.
    Args:
        algo: (str) Algorithm name.
        env: (str) Environment name.
    Returns:
        algo_args: (dict) Algorithm config.
        env_args: (dict) Environment config.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path = os.path.join(base_path, "configs", "envs_cfgs", f"{env}.yaml")

    with open(algo_cfg_path, "r", encoding="utf-8") as file:
        algo_args = yaml.load(file, Loader=yaml.FullLoader)
    with open(env_cfg_path, "r", encoding="utf-8") as file:
        env_args = yaml.load(file, Loader=yaml.FullLoader)
    return algo_args, env_args


def update_args(unparsed_dict, *args):
    """Update loaded config with unparsed command-line arguments.
    Args:
        unparsed_dict: (dict) Unparsed command-line arguments.
        *args: (list[dict]) argument dicts to be updated.
    """

    def update_dict(dict1, dict2):
        for k in dict2:
            if type(dict2[k]) is dict:
                update_dict(dict1, dict2[k])
            else:
                if k in dict1:
                    dict2[k] = dict1[k]

    for args_dict in args:
        update_dict(unparsed_dict, args_dict)


def get_task_name(env, env_args):
    """Get task name."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        if env_args["continuous_actions"]:
            task = f"{env_args['scenario']}-continuous"
        else:
            task = f"{env_args['scenario']}-discrete"
    elif env == "gym":
        task = env_args["scenario"]
    elif env == "football":
        task = env_args["env_name"]
    elif env == "dexhands":
        task = env_args["task"]
    elif env == "lag":
        task = f"{env_args['scenario']}-{env_args['task']}"
    elif env == "sokoban":
        task = env_args["scenario"]
    return task


def _format_run_value(value):
    """Format scalar hyperparameters compactly for directory names."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.001:
            return re.sub(r"e([+-])0+(\d+)", r"e\1\2", f"{value:.0e}")
        return f"{value:g}"
    return str(value)


def build_run_dir_name(prefix, algo_args, seed, timestamp=None):
    """Build a readable result directory name from key experiment settings."""
    timestamp = timestamp or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if not prefix:
        return "-".join(["seed-{:0>5}".format(seed), timestamp])

    train_args = algo_args.get("train", {})
    eval_args = algo_args.get("eval", {})
    model_args = algo_args.get("model", {})
    method_args = algo_args.get("algo", {})
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "auto").split(",")[0]

    parts = [
        prefix,
        f"steps{_format_run_value(train_args.get('num_env_steps', 'na'))}",
        f"len{_format_run_value(train_args.get('episode_length', 'na'))}",
        f"env{cuda_devices or 'auto'}",
        f"lr{_format_run_value(model_args.get('lr', 'na'))}",
        f"vlr{_format_run_value(model_args.get('critic_lr', 'na'))}",
    ]
    if "ppo_epoch" in method_args:
        parts.extend(
            [
                f"ppoepoch{_format_run_value(method_args['ppo_epoch'])}",
                f"clip{_format_run_value(method_args.get('clip_param', 'na'))}",
            ]
        )
    if "hasac" in prefix.lower():
        entropy_value = (
            "auto" if method_args.get("auto_alpha", False) else method_args.get("alpha", "na")
        )
        parts.append(f"entrocoe{_format_run_value(entropy_value)}")
    parts.extend(
        [
            f"evalepi{_format_run_value(eval_args.get('eval_episodes', 'na'))}",
            f"seed{_format_run_value(seed)}",
            timestamp,
        ]
    )
    return "-".join(parts)


def init_dir(
    env,
    env_args,
    algo,
    exp_name,
    seed,
    logger_path,
    algo_args=None,
    run_name_prefix="",
):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    run_dir_name = build_run_dir_name(run_name_prefix, algo_args or {}, seed)
    results_path = os.path.join(
        logger_path,
        env,
        task,
        algo,
        exp_name,
        run_dir_name,
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)

    try:
        from tensorboardX import SummaryWriter

        writter = SummaryWriter(log_path)
    except (ImportError, PermissionError, OSError):
        writter = DummySummaryWriter()
    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writter


def is_json_serializable(value):
    """Check if v is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except Error:
        return False


def convert_json(obj):
    """Convert obj to a version which can be serialized with JSON."""
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {
                convert_json(k): convert_json(v) for k, v in obj.__dict__.items()
            }
            return {str(obj): obj_dict}

        return str(obj)


def save_config(args, algo_args, env_args, run_dir):
    """Save the configuration of the program."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)


def prettify_json_file(path):
    """Rewrite a JSON file using readable indentation for humans and agents."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as src:
        payload = json.load(src)

    with open(path, "w", encoding="utf-8") as dst:
        json.dump(payload, dst, indent=2, ensure_ascii=False)
        dst.write("\n")
