import torch
import random
import os
import models
import argparse
import pandas as pd
import pickle
import json
import platform
import sys
import time
from dataset import setup 
from utils.model import train,test,save_model,save_performance,print_performance
from utils.io import parse_grid_parameters
from utils.generic import set_seed
from utils.params import Params


def write_reproducibility_metadata(params, model, train_seconds, test_seconds):
    """Persist run-time evidence required for reproducibility reporting."""
    output_file = getattr(params, 'output_file', '')
    if not output_file:
        return
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    device = params.device
    gpu = None
    peak_gpu_mib = None
    if device.type == 'cuda':
        gpu = {
            'index': int(device.index if device.index is not None else torch.cuda.current_device()),
            'name': torch.cuda.get_device_name(device),
            'cuda_runtime': torch.version.cuda,
        }
        peak_gpu_mib = float(torch.cuda.max_memory_allocated(device) / 2**20)
    test_profile = getattr(params, 'inference_profile', {}).get('test', {})
    batches = int(test_profile.get('batches', 0))
    examples = int(test_profile.get('examples', 0))
    forward_seconds = float(test_profile.get('forward_seconds', 0.0))
    metadata = {
        'config_file': str(getattr(params, 'config_file', '')),
        'dataset': params.dataset_name,
        'network_type': params.network_type,
        'seed': getattr(params, 'seed', None),
        'hyperparameters': {key: str(value) for key, value in params.__dict__.items()
                            if key not in {'reader', 'lookup_table', 'device', 'inference_profile', 'training_epoch_seconds'}},
        'parameters_total': int(sum(parameter.numel() for parameter in model.parameters())),
        'parameters_trainable': int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        'training_wall_seconds': float(train_seconds),
        'training_epoch_seconds': list(getattr(params, 'training_epoch_seconds', [])),
        'training_mean_epoch_seconds': float(sum(getattr(params, 'training_epoch_seconds', [])) / max(1, len(getattr(params, 'training_epoch_seconds', [])))),
        'test_evaluation_wall_seconds': float(test_seconds),
        'test_inference_batches': batches,
        'test_inference_examples': examples,
        'test_inference_forward_seconds': forward_seconds,
        'test_inference_batch_ms': 1000 * forward_seconds / batches if batches else None,
        'test_inference_example_ms': 1000 * forward_seconds / examples if examples else None,
        'peak_gpu_memory_mib': peak_gpu_mib,
        'hardware': {'cpu': platform.processor() or platform.uname().processor, 'gpu': gpu},
        'software': {'python': sys.version, 'pytorch': torch.__version__, 'platform': platform.platform()},
    }
    metadata_path = os.path.join(output_dir, 'reproducibility.json')
    with open(metadata_path, 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print('REPRODUCIBILITY_METADATA ' + json.dumps(metadata, ensure_ascii=False, sort_keys=True))

def run(params):   
    model = None
    if 'resume_model_file' in params.__dict__ and params.resume_model_file:
        print('Resuming training from checkpoint: {}'.format(params.resume_model_file))
        model = torch.load(params.resume_model_file, weights_only=False).to(params.device)
    elif 'load_model_from_dir' in params.__dict__ and params.load_model_from_dir:
        print('Loading the model from an existing dir!')
        model_params = pickle.load(open(os.path.join(params.dir_name,'config.pkl'),'rb'))
        if 'lookup_table' in params.__dict__:
            model_params.lookup_table = params.lookup_table
        if 'sentiment_dic' in params.__dict__:
            model_params.sentiment_dic = params.sentiment_dic
        model = models.setup(model_params)
        model.load_state_dict(torch.load(os.path.join(params.dir_name,'model')))
        model = model.to(params.device)
    else:
        model = models.setup(params).to(params.device)
      
    if params.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(params.device)
    params.inference_profile = {}
    train_started = time.perf_counter()
    if not ('fine_tune' in params.__dict__ and params.fine_tune == False):
        print('Training the model!')
        train(params, model)
        model = torch.load(params.best_model_file, weights_only=False)
        if 'retain_model_file' in params.__dict__ and params.retain_model_file:
            retain_dir = os.path.dirname(params.retain_model_file)
            if retain_dir:
                os.makedirs(retain_dir, exist_ok=True)
            torch.save(model, params.retain_model_file)
        os.remove(params.best_model_file)
        if ('training_checkpoint_file' in params.__dict__ and
                os.path.exists(params.training_checkpoint_file)):
            os.remove(params.training_checkpoint_file)
    train_seconds = time.perf_counter() - train_started
    test_started = time.perf_counter()
    performance_dict = test(model, params)
    write_reproducibility_metadata(params, model, train_seconds, time.perf_counter() - test_started)
    # performance_str = print_performance(performance_dict, params)
    # save_model(model,params,performance_str)
  
    return performance_dict

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/run.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    mode = 'run'
    if 'mode' in params.__dict__:
        mode = params.mode
    set_seed(params)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if mode == 'run':
        results = []
        reader = setup(params)
        reader.read(params)
        params.reader = reader   
        performance_dict = run(params)
        save_performance(params, performance_dict)
       
    elif mode == 'run_grid_search':
            
        print('Grid Search Begins.')
        if not 'grid_parameters_file' in params.__dict__:
            params.grid_parameters_file = params.network_type+'.ini'
            
        grid_parameters = parse_grid_parameters(os.path.join('config','grid_parameters',params.grid_parameters_file))
        df = pd.DataFrame()
        if not 'output_file' in params.__dict__:
            params.output_file = 'eval/grid_search_{}_{}.csv'.format(params.dataset_name, params.network_type)
        for i in range(params.search_times):
            parameter_list = []
            merged_dict = {}
            for key in grid_parameters:
                value = random.choice(grid_parameters[key])
                parameter_list.append((key, value))
                merged_dict[key] = value
            print(parameter_list)
            params.setup(parameter_list)
            reader = setup(params)
            reader.read(params)
            params.reader = reader
            performance_dict = run(params)
            performance_dict['model_dir_name'] = params.dir_name
            merged_dict.update(performance_dict)
            df = df.append(merged_dict, ignore_index=True)
            print(df)
            df.to_csv(params.output_file, encoding='utf-8', index=True)
    else:
        print('wrong input run mode!')
        exit(1)
        
        
        
   
