from torch.utils.data import DataLoader
import onnxruntime as ort
import torchvision
import numpy as np
import argparse
import torch
import tqdm
import onnx
import os


BENCHMARK_DIR_PATH = os.path.join(os.path.dirname(__file__), 'instances')
NETWORK_DIR_PATH = os.path.join(os.path.dirname(__file__), 'networks')
DATASET_DIR_PATH = os.path.join(os.path.dirname(__file__), 'datasets')
VERIFIER_DIR_PATH = os.path.join(os.path.dirname(__file__), 'tools')
TEMP_DIR_PATH = os.path.join(os.path.dirname(__file__), 'temp')
MAX_COUNT = 3 # number of instances per network
TIMEOUT = (6 * 3600) / (12 * MAX_COUNT)
EASY_INSTANCE_TIMEOUT = 20

NEURALSAT_PYTHON = os.getenv('NEURALSAT_PY', '')
CROWN_PYTHON = os.getenv('CROWN_PY', '')
if (not NEURALSAT_PYTHON) or (not CROWN_PYTHON):
    print('[!] Please run "source ./setup.sh" before running this script.')
    exit(1)


class ReturnStatus:

    UNSAT   = 'unsat'
    SAT     = 'sat'
    UNKNOWN = 'unknown'
    TIMEOUT = 'timeout'
    RESTART = 'restart'
    ERROR   = 'error'



CIFAR10_DATASET = torchvision.datasets.CIFAR10(
    root=f'{DATASET_DIR_PATH}/cifar10',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True,
)

def recursive_walk(rootdir):
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)
            

def inference_onnx(sess: ort.InferenceSession, *inputs: np.ndarray) -> list[np.ndarray]:
    names = [i.name for i in sess.get_inputs()]
    output = sess.run(None, dict(zip(names, inputs)))[0]
    return torch.from_numpy(output)


def _write_vnnlib(prefix:str, center: torch.Tensor, radius: float, prediction: torch.Tensor, 
                  data_lb: float, data_ub: float, dir_path: str, negate_spec=False, seed: int = 0) -> str:
    # output name
    spec_path = os.path.join(dir_path, f"{prefix}_eps_{radius:.5f}.vnnlib")

    # input bounds
    x_lb = torch.clamp(center - radius, min=data_lb, max=data_ub).flatten()
    x_ub = torch.clamp(center + radius, min=data_lb, max=data_ub).flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    with open(spec_path, "w") as f:
        f.write(f"; Spec for {seed=} {radius=:.5f}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        if negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
            f.write(f"))\n")
    return spec_path


def _get_dataloader(dataset: str):

    if dataset == 'cifar10':
        dataloader = DataLoader(CIFAR10_DATASET, batch_size=1, shuffle=True)
    else:
        raise NotImplementedError()
    return dataloader


def _generate_instance_per_dataset(dataset: str, fp, seed: int = 0):
    "Generate DNNV instances (net + spec) for specific dataset"
    dataloader = _get_dataloader(dataset)
    
    ort_sessions = {
        f: ort.InferenceSession(onnx.load(f).SerializeToString()) 
            for f in recursive_walk(f'{NETWORK_DIR_PATH}/{dataset}')
    }

    for net_path, session in ort_sessions.items():
        _generate_instance_per_network(net_path, session, dataloader, fp, seed=seed)


def _generate_instance_per_network(net_path, session, dataloader, fp, seed: int = 0):
    "Generate DNNV instances (net + spec) for specific network"
    # find image
    pbar = tqdm.tqdm(dataloader, desc=f'Generating specs for {os.path.basename(net_path)}')
    count = 0
    for i, (x, y) in enumerate(pbar):
        # get output
        pred = inference_onnx(session, x.numpy())
        
        # skip incorrect prediction sample
        if pred.argmax(-1) != y: 
            continue 
        
        # find epsilon
        for eps in np.linspace(0.01, 0.05, 21):
            # gen spec
            spec_path = _write_vnnlib(
                prefix=f'spec_{os.path.basename(net_path)[:-5]}_idx_{i}',
                center=x,
                radius=eps,
                prediction=pred,
                data_lb=0.0,
                data_ub=1.0,
                dir_path=TEMP_DIR_PATH, 
                seed=seed,
            )
            
            # filter easy instance
            inst_stat, inst_filter = _filter_instance(net_path, spec_path, EASY_INSTANCE_TIMEOUT)
            if inst_filter:
                if inst_stat == ReturnStatus.SAT:
                    print(f'Skip from eps={eps} due to a cex is found')
                    break
                continue
            
            # save
            print(net_path)
            print(spec_path) 
            _write_instance(net_path, spec_path, fp)
            
            # stat
            count += 1
            pbar.set_postfix(count=count)
            if count == MAX_COUNT:
                return
            
def _write_instance(net_path, spec_path, fp):
    os.system(f'cp {net_path} {BENCHMARK_DIR_PATH}/onnx/')
    os.system(f'cp {spec_path} {BENCHMARK_DIR_PATH}/vnnlib/')
    line = f'onnx/{os.path.basename(net_path)},vnnlib/{os.path.basename(spec_path)},{TIMEOUT}'
    # print('[+] Exported:', line)
    print(line, file=fp)
    

def _filter_instance(net_path, spec_path, timeout):
    inst_stat, inst_filter = _filter_instance_crown(net_path, spec_path, timeout)
    if inst_filter:
        return inst_stat, inst_filter
    
    inst_stat, inst_filter = _filter_instance_neuralsat(net_path, spec_path, timeout)
    if inst_filter:
        return inst_stat, inst_filter
    
    return inst_stat, False

def _handle_verifier_output(output):
    if output is None: # error
        return ReturnStatus.ERROR, True
    
    if 'unsat' in output.lower(): # easy unsat
        return ReturnStatus.UNSAT, True 
    
    if 'sat' in output.lower(): # easy sat
        return ReturnStatus.SAT, True 
    
    if 'timeout' in output.lower(): # not easy
        return ReturnStatus.TIMEOUT, False
    
    return ReturnStatus.UNKNOWN, True # unknown

def _filter_instance_neuralsat(net_path, spec_path, timeout):
    "Filter out easy instances"
    output = _run_neuralsat(net_path, spec_path, timeout)
    return _handle_verifier_output(output)


def _filter_instance_crown(net_path, spec_path, timeout):
    "Filter out easy instances"
    output = _run_crown(net_path, spec_path, timeout)
    return _handle_verifier_output(output)



def _run_neuralsat(net_path, spec_path, timeout):
    res_file = f'{TEMP_DIR_PATH}/neuralsat_res.txt'
    os.system(f'rm -rf {res_file}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmd = f'{NEURALSAT_PYTHON} {VERIFIER_DIR_PATH}/neuralsat/neuralsat-pt201/main.py --net {net_path} --spec {spec_path} --timeout {timeout} --device {device} --result_file {res_file} > /dev/null 2>&1'
    # print(cmd)
    os.system(cmd)
    
    if not os.path.exists(res_file):
        return None
    
    output = open(res_file).read().strip()
    os.system(f'rm -rf {res_file}')
    return output


def _run_crown(net_path, spec_path, timeout):
    res_file = f'{TEMP_DIR_PATH}/abcrown_res.txt'
    os.system(f'rm -rf {res_file}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = f'{VERIFIER_DIR_PATH}/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/cifar2020_2_255.yaml' # default config
    cmd = f'{CROWN_PYTHON} {VERIFIER_DIR_PATH}/alpha-beta-CROWN/complete_verifier/abcrown.py --config {config} --onnx_path {net_path} --vnnlib_path {spec_path} --timeout {timeout} --device {device} --results_file {res_file}> /dev/null 2>&1'
    # print(cmd)
    os.system(cmd)
    
    if not os.path.exists(res_file):
        return None
    
    output = open(res_file).read().strip()
    os.system(f'rm -rf {res_file}')
    return output
    

def generate(args):
    torch.manual_seed(args.seed)
    os.makedirs(f'{BENCHMARK_DIR_PATH}/onnx', exist_ok=True)
    os.makedirs(f'{BENCHMARK_DIR_PATH}/vnnlib', exist_ok=True)
    os.makedirs(TEMP_DIR_PATH, exist_ok=True)
    
    with open(f'{BENCHMARK_DIR_PATH}/instances.csv', 'w') as fp:
        for dataset in os.listdir(NETWORK_DIR_PATH):
            _generate_instance_per_dataset(dataset, fp, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description='Benchmark generator',)
    parser.add_argument('seed', type=int, help='Random seed for generation')
    args = parser.parse_args()
        
    generate(args)
    
    
if __name__ == "__main__":
    main()
    