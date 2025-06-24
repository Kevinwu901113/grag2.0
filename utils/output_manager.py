import os
import datetime

def get_latest_work_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return None
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("grag_")]
    if not subdirs:
        return None
    subdirs.sort()
    return os.path.join(base_dir, subdirs[-1])

def create_new_work_dir(base_dir: str) -> str:
    run_id = datetime.datetime.now().strftime("grag_%Y%m%d_%H%M")
    work_dir = os.path.join(base_dir, run_id)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir

def resolve_work_dir(config: dict, args) -> str:
    base_dir = config["output"]["work_dir"]

    if hasattr(args, 'work_dir') and args.work_dir:
        return args.work_dir
    elif hasattr(args, 'new') and args.new:
        return create_new_work_dir(base_dir)
    else:
        last = get_latest_work_dir(base_dir)
        return last if last else create_new_work_dir(base_dir)
