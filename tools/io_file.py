import os
import json
import csv
import yaml
import pandas as pd
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor

from tools import logger_factory

logger = logger_factory.setup_logger('io_file')

def create_if_not_exist(path):
    if not os.path.exists(os.path.dirname(path)): 
        os.makedirs(path, exist_ok=True) 


def read(file_path, chunk_size=1024*1024*10000, line_threshold=1000000, encoding='utf-8'):
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.txt':
            return read_txt(file_path, chunk_size, encoding)
        elif file_ext == '.csv':
            return read_csv(file_path, chunk_size, encoding)
        elif file_ext == '.json':
            return read_json(file_path, encoding)
        elif file_ext == '.yaml' or file_ext == '.yml':
            return read_yaml(file_path, encoding)
        elif file_ext == '.xlsx':
            return read_xlsx(file_path, encoding)
        elif file_ext == '.md':
            return read_markdown(file_path, encoding)
        elif file_ext == '.jsonl':
            return read_jsonl(file_path, chunk_size, line_threshold, encoding)
        else:
            raise ValueError(f"unsupport data format: {file_ext}")
    except Exception as e:
        logger.error(f"read {file_path} error: {e}")
        raise


def write(file_path, data, mode='a', chunk_size=1024*1024*100000, encoding='utf-8'):

    create_if_not_exist(file_path)
        
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.txt':
            write_txt(file_path, data, mode, chunk_size, encoding)
        elif file_ext == '.csv':
            write_csv(file_path, data, mode, chunk_size, encoding)
        elif file_ext == '.json':
            write_json(file_path, data, mode, encoding)
        elif file_ext == '.yaml' or file_ext == '.yml':
            write_yaml(file_path, data, mode, encoding)
        elif file_ext == '.xlsx':
            write_xlsx(file_path, data, encoding)
        elif file_ext == '.md':
            write_markdown(file_path, data, mode, encoding)
        elif file_ext == '.jsonl':
            write_jsonl(file_path, data, mode, chunk_size, encoding)
        else:
            raise ValueError(f"unsupport data format: {file_ext}")
    except Exception as e:
        logger.error(f"write {file_path} error: {e}")
        raise




def read_txt(file_path, chunk_size, encoding='utf-8'):
    file_size = os.path.getsize(file_path)
    if file_size > chunk_size:
        return read_txt_multithread(file_path, chunk_size, encoding)
    else:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()


def read_txt_multithread(file_path, chunk_size, encoding='utf-8'):

    file_size = os.path.getsize(file_path)
    num_chunks = (file_size // chunk_size) + 1

    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, num_chunks)

    chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(read_chunk, file_path, i * chunk_size, (i + 1) * chunk_size, encoding) for i in range(num_chunks)]
        for future in tqdm(futures, desc="reading text file progress"):
            chunks.append(future.result())

    return ''.join(chunks)


def read_chunk(file_path, start, end, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        f.seek(start)
        return f.read(end - start)
    

def read_csv(file_path, chunk_size, encoding='utf-8'):

    file_size = os.path.getsize(file_path)
    if file_size > chunk_size:
        return read_csv_multithread(file_path, chunk_size, encoding)
    else:
        return pd.read_csv(file_path, encoding=encoding)


def read_csv_multithread(file_path, chunk_size, encoding='utf-8'):
    file_size = os.path.getsize(file_path)
    num_chunks = (file_size // chunk_size) + 1

    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, num_chunks)

    data_chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(read_csv_chunk, file_path, i * chunk_size, (i + 1) * chunk_size, encoding) for i in range(num_chunks)]
        for future in tqdm(futures, desc="reading csv file progress"):
            data_chunks.append(future.result())

    return pd.concat(data_chunks, ignore_index=True)


def read_csv_chunk(file_path, start, end, encoding='utf-8'):
    return pd.read_csv(file_path, skiprows=range(1, start), nrows=end - start, encoding=encoding)


def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return json.load(f)

def read_yaml(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)


def read_xlsx(file_path, encoding='utf-8'):
    return pd.read_excel(file_path)


def read_markdown(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def read_jsonl(file_path, chunk_size, line_threshold, encoding='utf-8'):
    file_size = os.path.getsize(file_path)
    num_lines = sum(1 for _ in open(file_path, 'r', encoding=encoding))

    if file_size > chunk_size or num_lines > line_threshold:
        return read_jsonl_multithread(file_path, chunk_size, num_lines, encoding)
    else:
        with open(file_path, 'r', encoding=encoding) as f:
            return [json.loads(line) for line in f]


def read_jsonl_multithread(file_path, chunk_size, num_lines, encoding='utf-8'):
    chunks = []
    lines_per_chunk = num_lines // (os.cpu_count() // 2) 

    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, (num_lines // lines_per_chunk) + 1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(read_jsonl_chunk, file_path, i * lines_per_chunk, (i + 1) * lines_per_chunk, encoding)
            for i in range(num_threads)
        ]
        for future in tqdm(futures, desc="reading jsonl file progress"):
            chunks.append(future.result())

    return [item for sublist in chunks for item in sublist]


def read_jsonl_chunk(file_path, start_line, end_line, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        return [json.loads(line) for _, line in zip(range(start_line), f)][start_line:end_line]


def write_txt(file_path, data, mode, chunk_size, encoding='utf-8'):
    data_size = len(data)
    if data_size > chunk_size:
        write_txt_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        with open(file_path, mode, encoding=encoding) as f:
            f.write(data)


def write_txt_multithread(file_path, data, mode, chunk_size, encoding='utf-8'):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, len(chunks))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_chunk, file_path, chunk, mode, encoding) for chunk in chunks]
        for future in tqdm(futures, desc="Writing text file progress"):
            future.result()


def write_chunk(file_path, chunk, mode, encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        f.write(chunk)


def write_jsonl(file_path, data, mode, chunk_size, encoding='utf-8'):
    data_size = len(data)
    if data_size > chunk_size:
        write_jsonl_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        with open(file_path, mode, encoding=encoding) as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')


def write_jsonl_multithread(file_path, data, mode, chunk_size, encoding='utf-8'):
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, len(chunks))


    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_jsonl_chunk, file_path, chunk, mode, encoding) for chunk in chunks]
        for future in tqdm(futures, desc="Writing JSONL file progress"):
            future.result()


def write_jsonl_chunk(file_path, chunk, mode, encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        for entry in chunk:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def write_csv(file_path, data, mode, chunk_size, encoding='utf-8'):
    data_size = data.memory_usage(index=True).sum() 
    if data_size > chunk_size:
        write_csv_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        data.to_csv(file_path, mode=mode, encoding=encoding, index=False)


def write_csv_multithread(file_path, data, mode, chunk_size, encoding='utf-8'):
    chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2 
    num_threads = min(cpu_count, len(chunks))


    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_csv_chunk, file_path, chunk, mode, encoding) for chunk in chunks]
        for future in tqdm(futures, desc="Writing CSV file progress"):
            future.result()


def write_csv_chunk(file_path, chunk, mode, encoding='utf-8'):
    chunk.to_csv(file_path, mode=mode, encoding=encoding, index=False, header=(mode == 'w'))


def write_json(file_path, data, mode, encoding='utf-8'):
    if mode == 'a' and os.path.exists(file_path):
        with open(file_path, 'r+', encoding=encoding) as f:
            existing_data = json.load(f)
            existing_data.extend(data)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    else:
        with open(file_path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def write_yaml(file_path, data, mode, encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(data, f, allow_unicode=True)


def write_csv(file_path, data, mode, chunk_size, encoding='utf-8'):
    data_size = data.memory_usage(index=True).sum()
    if data_size > chunk_size:
        write_csv_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        data.to_csv(file_path, mode=mode, encoding=encoding, index=False)


def write_csv_multithread(file_path, data, mode, chunk_size, encoding='utf-8'):
    chunks = [data.iloc[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2
    num_threads = min(cpu_count, len(chunks))


    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_csv_chunk, file_path, chunk, mode, encoding) for chunk in chunks]
        for future in tqdm(futures, desc="Writing CSV file progress"):
            future.result()


def write_csv_chunk(file_path, chunk, mode, encoding='utf-8'):
    chunk.to_csv(file_path, mode=mode, encoding=encoding, index=False, header=(mode == 'w'))


def write_json(file_path, data, mode, encoding='utf-8'):
    if mode == 'a' and os.path.exists(file_path):
        with open(file_path, 'r+', encoding=encoding) as f:
            existing_data = json.load(f)
            existing_data.extend(data)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    else:
        with open(file_path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def write_yaml(file_path, data, mode, encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(data, f, allow_unicode=True)


def write_xlsx(file_path, data, encoding='utf-8'):
    data.to_excel(file_path, index=False)


def write_markdown(file_path, data, mode, encoding='utf-8'):
    with open(file_path, mode, encoding=encoding) as f:
        f.write(data)
