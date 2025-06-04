import sys
import os
import argparse
import os
import json
from datasets import load_dataset
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import utils
from tools.logger_factory import setup_logger

logger = setup_logger("build_predict_path_dataset")


def build_data(args):
    '''
    Extract the paths between question and answer entities from the dataset.
    '''
    
    input_file = os.path.join(args.data_path, args.dataset)
    output_dir = os.path.join(args.output_path, args.dataset)
    output_path = os.path.join(output_dir, args.save_name)
    
    logger.info(f"Save results to: {output_path}")
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    
    # Load dataset
    dataset = load_dataset(input_file, split=args.split)
    with open(output_path, 'w') as fout:
        with mp.Pool(args.process_num) as pool:
            for res in tqdm(pool.imap_unordered(partial(process_data, remove_duplicate=args.remove_duplicate, predict_type=args.predict_type), dataset), total=len(dataset)):
                for r in res:
                    fout.write(json.dumps(r) + '\n')


def process_data(data, remove_duplicate=False, predict_type="triple"):
    question = data['question']
    graph  =  utils.build_graph(data['graph'])
    paths = utils.get_truth_paths(data['q_entity'], data['a_entity'], graph)
    result = []

    rel_paths = []
    for path in paths:
        if predict_type == "triple":
            rel_path = [p for p in path] # extract triple path
        elif predict_type == "relation":
            rel_path = [p[1] for p in path] # extract relation path
        if remove_duplicate:
            if tuple(rel_path) in rel_paths:
                continue
        rel_paths.append(tuple(rel_path))
    for rel_path in rel_paths:
        result.append({"question": question, "path": rel_path})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument('--dataset', type=str, default='webqsp')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument("--output_path", type=str, default="data/webqsp/predict_path")
    parser.add_argument("--predict_type", type=str, default="triple", help="triple path or relation path. Choice: 'triple', 'relation' ")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument('--process_num', '-n', type=int, default=16)
    parser.add_argument('--remove_duplicate', action='store_true')
    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.dataset + "_" + args.split + "_" + args.predict_type + ".jsonl"
    
    build_data(args)
