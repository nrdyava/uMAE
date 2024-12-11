import os
import argparse
from datetime import datetime
import pytz
import yaml
import copy
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configuration file path for the task ', dest='config_file')
    args = parser.parse_args()
    return args

 
def start_of_a_run():
    args = parse_args()
    config = copy.deepcopy(yaml.safe_load(open(args.config_file, 'r')))
    
    run_start_time_utc = datetime.now(pytz.utc)
    run_start_time_local = run_start_time_utc.astimezone(pytz.timezone(config["local_time_zone"])).strftime("%Y-%m-%d-%H-%M-%S-%f")
    run_name = run_start_time_local + ' + ' + config['exp_name']
    config['run_name'] = run_name
    config['wandb_name'] = run_name
    
    runs_dir = config["runs_dir"]
    run_dir = os.path.join(runs_dir, run_name)
    config['run_dir'] = run_dir
    
    start_of_a_run_rank_zero(args, config, run_start_time_local, run_dir)
    return config


@rank_zero_only
def start_of_a_run_rank_zero(args, config, run_start_time_local, run_dir):
    # create a directory to store the results of the run
    os.mkdir(config['run_dir'])
    
    # save the configuration file in the run directory. Useful to later check the configuration used for the run.
    yaml.dump(config, open(os.path.join(config['run_dir'], 'config.yaml'), 'w'), default_flow_style=False, sort_keys=False)