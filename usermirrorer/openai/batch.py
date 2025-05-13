import os
import json
import openai
import logging
import time
import numpy as np
import pandas as pd
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

from .openai import run_batch_instance, batch_create

logger = logging.getLogger("rich")


def run_batch_vllm(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    client: openai.OpenAI | None = None,
    **kwargs
) -> None:
    """
    Run batch inference on VLLM model.
    """

    model_path = kwargs.get("model_path", model)
    chunksize = kwargs.get("chunksize", 50000)
    data_parallel_size = kwargs.get("data_parallel_size", 2)
    gpu_ids = kwargs.get("gpu_ids", "0").split(",")

    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"
    
    command = f"""python -m vllm.entrypoints.openai.run_batch  \
    --model {model_path} \
    --served-model-name {model} \
    --trust-remote-code \
    --tensor-parallel-size {kwargs.get("tensor_parallel_size", 1)} \
    --gpu-memory-utilization {kwargs.get("gpu_memory_utilization", 0.9)} \
    --max-model-len {kwargs.get("max_model_len", 8192)} \
    --disable-log-requests \
    --dtype auto \
    --seed 42 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    """

    if chunksize:
        open(output_file, "w").close()
        
        # Create task queue of chunks
        task_queue = []
        for ind, input_chunk in enumerate(pd.read_json(input_file, lines=True, chunksize=chunksize)):
            input_chunk_file = input_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
            output_chunk_file = output_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
            input_chunk.to_json(input_chunk_file, orient='records', lines=True)
            task_queue.append((ind, input_chunk_file, output_chunk_file))

        # Process chunks in parallel across GPUs
        with ThreadPoolExecutor(max_workers=data_parallel_size) as executor:
            futures = []
            for gpu_id in range(data_parallel_size):
                gpu_tasks = task_queue[gpu_id::data_parallel_size]
                if not gpu_tasks:
                    continue
                    
                def process_gpu_tasks(tasks, gpu_id):
                    for _, input_file, output_file in tasks:
                        gpu_command = f"CUDA_VISIBLE_DEVICES={gpu_ids[gpu_id]} " + command + \
                                    f"--input-file {input_file} --output-file {output_file}"
                        retry_count = 0
                        while retry_count < 3:
                            try:
                                print(gpu_command)
                                os.system(gpu_command)      
                                pd.read_json(output_file, lines=True).to_json(
                                    output_file.replace("_chunk-", "_processed-"), 
                                    orient='records', lines=True
                                )
                                break
                            except Exception as e:
                                retry_count += 1
                                if retry_count == 3:
                                    raise e
                                with open("log/error.log", "a") as f:
                                    f.write(f"Error processing {output_file}: {e}\n. Retry {retry_count}/3\n")
   
                        os.remove(input_file)
                        os.remove(output_file)
                
                futures.append(executor.submit(process_gpu_tasks, gpu_tasks, gpu_id))
                time.sleep(10)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()
                
        # Combine all processed chunks
        processed_files = sorted([f for f in os.listdir(os.path.dirname(output_file)) 
                                if "_processed-" in f])
        for f in processed_files:
            file_path = os.path.join(os.path.dirname(output_file), f)
            pd.read_json(file_path, lines=True).to_json(
                output_file, orient='records', lines=True, mode='a'
            )
            os.remove(file_path)
    else:
        os.system(command + f"--input-file {input_file} --output-file {output_file}")

def run_batch_openai(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    client: openai.OpenAI | None = None,
    max_running_task: int = 3,
    **kwargs
) -> None:
    """
    Run batch inference on OpenAI API batch mode.
    Args:
        input_file: Input file path
        output_file: Output file path
        model: Model name
        client: OpenAI client
        max_running_task: Maximum number of concurrent running tasks
        **kwargs: Additional arguments
    """
    chunksize = kwargs.get("chunksize", 4096)

    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"
    
    # Read all chunks first
    chunks = []
    for ind, input_chunk in enumerate(pd.read_json(input_file, lines=True, chunksize=chunksize)):
        input_chunk_file = input_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
        output_chunk_file = output_file.replace(".jsonl", f"_chunk-{ind}.jsonl")
        input_chunk.to_json(input_chunk_file, orient='records', lines=True)
        chunks.append((ind, input_chunk_file, output_chunk_file))

    task_ids = {}
    pending_chunks = chunks.copy()
    
    def submit_task(chunk_info):
        ind, input_chunk_file, output_chunk_file = chunk_info
        task_id = batch_create(client, input_chunk_file)
        os.remove(input_chunk_file)
        task_ids[task_id] = {
            "status": False,
            "output_file": output_chunk_file
        }
        print(f"Task {task_id} created")
        return task_id

    # Submit initial batch of tasks up to max_running_task
    initial_tasks = pending_chunks[:max_running_task]
    pending_chunks = pending_chunks[max_running_task:]
    for chunk_info in initial_tasks:
        submit_task(chunk_info)
    
    running_time = time.time()

    while True:
        completed_tasks = []
        for task_id in task_ids:
            if task_ids[task_id]["status"]:
                continue
            
            task = client.batches.retrieve(task_id)
            if task.status in ["validating", "in_progress", "finalizing"]:
                pass
            elif task.status == "completed":
                file_response = client.files.content(task.output_file_id)
                with open(task_ids[task_id]["output_file"], "wb") as f:
                    f.write(file_response.content)
                task_ids[task_id]["status"] = True
                completed_tasks.append(task_id)
            else:
                raise Exception(f"Task {task_id} failed with status {task.status}")
        
        # Submit new tasks for each completed task if there are pending chunks
        for _ in completed_tasks:
            if pending_chunks:
                chunk_info = pending_chunks.pop(0)
                submit_task(chunk_info)
        
        if all([task_ids[task_id]["status"] for task_id in task_ids]) and not pending_chunks:
            break

        num_finished = sum([task_ids[task_id]["status"] for task_id in task_ids])
        num_running = len(task_ids) - num_finished
        print(f"{num_finished} tasks finished, {num_running} tasks running, {len(pending_chunks)} tasks pending. Time taken: {round((time.time() - running_time)/60, 1)} minutes")
        sleep(kwargs.get("update_interval", 60))
    
    print("All tasks finished, concatenating results...")
    open(output_file, "w").close()  # Clear the file
    for task_id in task_ids:
        pd.read_json(task_ids[task_id]["output_file"], lines=True).to_json(output_file, orient='records', lines=True, mode='a')
        os.remove(task_ids[task_id]["output_file"])
    print(f"""Results concatenated to "{output_file}" """)

  
def run_batch_instant_api(
    input_file: str,
    output_file: str | None = None,
    model: str | None = "google/Gemma-2-2B-it",
    client: openai.OpenAI | None = None,
    **kwargs
) -> None:
    """
    Run batch inference on OpenAI API instant mode.
    """

    parallel_size = kwargs.get("parallel_size", 1)
    if not output_file:
        output_file = input_file.rsplit(".", 1)[0] + "_results.jsonl"

    output = open(output_file, "a")

    def process_line(line):
        row = json.loads(line)
        result = run_batch_instance(row, client)
        return json.dumps(result) + "\n"

    with open(input_file, "r") as f:
        lines = f.readlines()

    if parallel_size > 1:
        with ThreadPoolExecutor(max_workers=parallel_size) as executor:
            futures = [executor.submit(process_line, line) for line in lines]
            for future in as_completed(futures):
                output.write(future.result())
    else:
        for line in lines:
            output.write(process_line(line))
    
    output.close()
