from dataclasses import dataclass
import subprocess
from subprocess import CompletedProcess
from typing import List
from enum import Enum
import torch
import os
import time
import signal
import csv
import argparse
import json
from pathlib import Path

# from analyse import analyse_results

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROFILING_DATA_DIR = PROJECT_ROOT_DIR / "data"


# TODO: for automation, load these from a json file
@dataclass
class BenchSetting:
    model: str
    tp: int
    device: str
    dataset: str
    req_rate: float
    num_requests: int
    port: int = 10000
    speculative_model: str = None
    num_speculative_tokens: int = -1
    speculative_draft_tensor_parallel_size: int = -1

    @staticmethod
    def get_head():
        return [
            "model",
            "tp",
            "device",
            "dataset",
            "req_rate",
            "num_requests",
            "speculative_model",
            "num_speculative_tokens",
            "speculative_draft_tensor_parallel_size",
        ]

    def get_value_list(self):
        return [
            self.model,
            self.tp,
            self.device,
            self.dataset,
            self.req_rate,
            self.num_requests,
            self.speculative_model,
            self.num_speculative_tokens,
            self.speculative_draft_tensor_parallel_size,
        ]


@dataclass
class BenchResult:
    setting: BenchSetting
    mean_TTFT: float
    mean_TPOT: float
    p99_TTFT: float
    p99_TPOT: float
    mean_e2e_latency: float
    p99_e2e_latency: float
    total_time: float
    throughput: float

    @staticmethod
    def get_head():
        return BenchSetting.get_head() + [
            "mean_TTFT",
            "mean_TPOT",
            "p99_TTFT",
            "p99_TPOT",
            "mean_e2e_latency",
            "p99_e2e_latency",
            "total_time",
            "throughput",
        ]

    def get_value_list(self):
        return self.setting.get_value_list() + [
            self.mean_TTFT,
            self.mean_TPOT,
            self.p99_TTFT,
            self.p99_TPOT,
            self.mean_e2e_latency,
            self.p99_e2e_latency,
            self.total_time,
            self.throughput,
        ]


class Util:
    @staticmethod
    def run_cmd(cmd, blocking=True):
        def set_new_pgroup():
            os.setpgrp()

        print(cmd)
        if blocking:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            return subprocess.Popen(cmd, shell=True, preexec_fn=set_new_pgroup)


class BenchEngine:
    def __init__(self, setting: BenchSetting) -> None:
        self.backend_process: subprocess.Popen = None
        # launch backend
        cmd = (
            f"vllm serve {setting.model} "
            f" --tensor-parallel-size {setting.tp} "
            f" --disable-log-requests"
            f" --max-model-len 40960"
            f" --port {setting.port}"
            f" --enable-chunked-prefill=False"
        )
        if setting.speculative_model:
            cmd = "VLLM_USE_FLASHINFER_SAMPLER=1 " + cmd
            cmd += f" --speculative-model {setting.speculative_model}"
            cmd += " --use-v2-block-manager"
        if setting.num_speculative_tokens >= 0:
            cmd += f" --num-speculative-tokens {setting.num_speculative_tokens}"
        if setting.speculative_draft_tensor_parallel_size > 0:
            cmd += f" --speculative-draft-tensor-parallel-size {setting.speculative_draft_tensor_parallel_size}"
        if setting.speculative_model == "[ngram]":
            cmd += f" --ngram-prompt-lookup-max {args.ngram_prompt_lookup_max}"
            cmd += f" --ngram-prompt-lookup-min {args.ngram_prompt_lookup_min}"

        self.backend_process = Util.run_cmd(cmd, False)

    def bench(self, runs: List[BenchSetting]) -> List[BenchResult]:
        time.sleep(120)
        print("============Start Benchmarking==================")
        return [self.bench_single(run) for run in runs]

    def bench_single(self, run: BenchSetting) -> BenchResult:
        out_values = [str(x) for x in run.get_value_list()]
        out_values = [x.replace("/", "_") for x in out_values]
        result_filename = f"bench_results/{':'.join(out_values)}.json"
        cmd = (
            f"python benchmarks/benchmark_serving.py "
            f" --model {run.model} --request-rate {run.req_rate}"
            f" --num-prompts {run.num_requests}"
            f" --save-result "
            f" --result-filename {result_filename}"
            f" --port {run.port}"
        )
        if run.dataset == "cnn_dailymail":
            cmd += f" --dataset-name {run.dataset}"
        else:
            cmd += f" --dataset {run.dataset}"
        completed_process: CompletedProcess = Util.run_cmd(cmd, True)

        def process_output(completed_process: CompletedProcess):
            if completed_process.returncode != 0:
                print(f"[Error] {completed_process.stdout}")
                print(f"[Error] {completed_process.stderr}")
                return BenchResult(run, -1, -1, -1, -1, -1, -1, -1, -1)
            with open(result_filename, "r") as f:
                result = json.load(f)
                return BenchResult(
                    run,
                    result["mean_ttft_ms"],
                    result["mean_tpot_ms"],
                    result["p99_ttft_ms"],
                    result["p99_tpot_ms"],
                    result["mean_e2el_ms"],
                    result["p99_e2el_ms"],
                    result["duration"],
                    result["output_throughput"],
                )

        return process_output(completed_process)

    def dump_results(self, results: List[BenchResult], outfile: str) -> None:
        with open(outfile, "a") as f:
            writer = csv.writer(f)
            writer.writerow(BenchResult.get_head())
            for result in results:
                writer.writerow(result.get_value_list())

    def __del__(self):
        # stop backend
        print("==============Finish Benchmarking==============")
        if (
            self.backend_process.poll() is None
        ):  # If poll() returns None, the process is still running
            print("Process is running, trying to kill...")
            os.killpg(self.backend_process.pid, signal.SIGINT)
            time.sleep(10)  # wait a bit for cleaning resources
            self.backend_process.terminate()
            self.backend_process.wait()
            time.sleep(1)
            if self.backend_process.poll() is not None:
                print(f"Process {self.backend_process.pid} killed successfully.")
            else:
                print("Failed to kill process.")
        else:
            print("Process already terminated.")


def main(args):
    device = torch.cuda.get_device_name(0).replace(" ", "_")
    request_rate_start, request_rate_end, step = args.request_rate_params

    runs = []
    request_rates = []
    # All * 10 to generate non-integer request rates
    bench_setting = None
    if "70" in args.model:
        tp = 4
    else:
        tp = 1

    if args.speculative_model and "7B" not in args.speculative_model:
        speculative_draft_tensor_parallel_size = 1
    else:
        speculative_draft_tensor_parallel_size = -1

    for req_rate in range(
        request_rate_start * 10, (request_rate_end + step) * 10, step * 10
    ):
        req_rate = req_rate / 10.0
        request_rates.append(req_rate)
        bench_setting = BenchSetting(
            args.model,
            tp,
            device,
            args.dataset,
            req_rate,
            args.num_requests,
            10000,  # port
            args.speculative_model,
            args.num_speculative_tokens,
            speculative_draft_tensor_parallel_size,
        )
        runs.append(bench_setting)
    engine = BenchEngine(bench_setting)
    results = engine.bench(runs)
    engine.dump_results(results, "bench_results/all_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=-1)
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=-1)
    parser.add_argument("--num-speculative-tokens", type=int, default=-1)

    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument("--outfile", type=str, default="bench_results")
    parser.add_argument("--num_requests", type=int, default=200)
    parser.add_argument(
        "--request_rate_params",
        type=tuple,
        help="(start_request_rate, end_request_rate, step_size). End_request_size is INCLUDED.",
        default=(1, 10, 1),
    )
    args = parser.parse_args()

    main(args)
