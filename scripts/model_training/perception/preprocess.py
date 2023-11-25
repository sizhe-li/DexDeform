import argparse
import gc

from policy.preprocess.preprocess_plab import (
    Preprocessor,
    preprocess_plab_demos_to_scenes,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="Removing old output directory"
    )
    parser.add_argument(
        "--remove_exists", action="store_true", help="Removing old output directory"
    )
    parser.add_argument("--env_name", type=str, default=None)

    args = parser.parse_args()
    return args


def preproc_files(env_name, files, remove_exists=False, debug=False):
    proc = Preprocessor(remove_exists=remove_exists, debug=debug)

    proc.init_env(env_name)
    proc.set_demo_paths(files)
    proc.preprocess()

    del proc
    gc.collect()


if __name__ == "__main__":
    args = get_args()

    preprocess_plab_demos_to_scenes(
        remove_exists=args.remove_exists,
        debug=args.debug,
        env_name=args.env_name,
    )
