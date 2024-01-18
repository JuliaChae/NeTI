import sys

import pyrallis
import argparse

from diffusers.utils import check_min_version

sys.path.append(".")
sys.path.append("..")

from training.coach import Coach
from training.config import RunConfig
import utils.pidfile as pidfile

parser = argparse.ArgumentParser()

# Add your script's configuration parameters here
# parser.add_argument("--exp_name", type=str, default="exp_name", help="Experiment name")
# parser.add_argument("--exp_dir", type=str, default="/data/vision/beery/fgg_ai/output", help="Path to the output directory")
# parser.add_argument("--train_data_dir", type=str, default="", help="Path to the training data directory")
# parser.add_argument("--super_category_token", type=str, default="oriole", help="Broad token for training initialization")
# parser.add_argument("--num_training_imgs", type=int, default=10, help="Number of training images to use")
# parser.add_argument("--config_path", type=str, default="input_configs/train.yaml", help="Path to the config file")

# args, remaining_args = parser.parse_known_args()

# # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.14.0")

# # Load the Pyrallis configuration
# breakpoint()
# config = pyrallis.load_config(args.config_path)

# # Create a RunConfig instance
# run_config = RunConfig(config)

# # Merge the custom arguments into the RunConfig
# run_config.merge(args)

# breakpoint()
# #args = parser.parse_args()

# # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")


@pyrallis.wrap()
def main(cfg: RunConfig):
    #cfg.data.num_training_imgs = 10 #args.num_training_imgs
    prepare_directories(cfg=cfg)
    coach = Coach(cfg)
    
    pidfile.exit_if_job_done(cfg.log.exp_dir)
    coach.train()
    pidfile.mark_job_done(cfg.log.exp_dir)

def prepare_directories(cfg: RunConfig):
    cfg.log.exp_dir = cfg.log.exp_dir / cfg.log.exp_name
    cfg.log.exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.log.logging_dir = cfg.log.exp_dir / cfg.log.logging_dir
    cfg.log.logging_dir.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    main()
