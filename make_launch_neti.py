import os
import json
import argparse
import socket
import glob
import yaml

TRAIN_PATH = "/data/vision/beery/fgg_ai/inat/inat_train_filtered"
NUM_TRAINING_IMGS = [10]

# parser = argparse.ArgumentParser()
# parser.add_argument('--classify_config', type=str, default="configs/classify_config.yaml")
# parser.add_argument('--user_config', type=str, default="configs/julia_configs.yaml")
# args = parser.parse_args()

# with open(args.classify_config, 'r') as file:
#     config = yaml.safe_load(file)

# with open(args.user_config, 'r') as file:
#     user_config = yaml.safe_load(file)

def make_files(commands):
    bash_file = "launch/run_neti.sh"
    with open(bash_file, "w") as f:
        for i, command in enumerate(commands):
            if i != 0:
                f.write("\n")
            f.write(command)

def get_train_command(species, num_imgs):
    data_dir = os.path.join(TRAIN_PATH, species)
    class_id = species.split('_')[0]
    exp_name = f"neti_{class_id}_{num_imgs}_imgs"
    exp_dir = "/data/vision/beery/fgg_ai/output/neti_sd_1.5"
    command = f"""python scripts/train.py \
        --config_path input_configs/train.yaml \
        --log.exp_name={exp_name} \
        --log.exp_dir={exp_dir} \
        --data.train_data_dir={data_dir} \
        --data.super_category_token=bird \
        --data.num_training_imgs={num_imgs} \
        --model.mapper_output_dim=768"""
    return command

def get_inference_command(species, num_imgs):
    class_id = species.split('_')[0]
    exp_name = f"neti_{class_id}_{num_imgs}_imgs"
    seeds = list(range(0,100))
    exp_dir = "/data/vision/beery/fgg_ai/output/neti_sd_1.5"
    command = f"""python scripts/inference.py \
        --config_path input_configs/inference.yaml \
        --input_dir={os.path.join(exp_dir, exp_name)}"""
    return command

if __name__ == "__main__":
    commands = []
    directories = [d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))]
    for species in directories:
        for num_img in NUM_TRAINING_IMGS:
            # only write in command if num training images > folders 
            files = glob.glob(f"{os.path.join(TRAIN_PATH, species)}/*.jpg")
            if len(files) >= num_img:
                commands.append(get_train_command(species, num_img))
    make_files(commands)

