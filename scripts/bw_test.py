#!/usr/bin/env python3

import argparse
import csv
import gc
import logging
import os
import sys
import time

import numpy as np
import torch
import zmq
from mpi4py import MPI

from diflow.backend.data_engine.engine.nvshmem_data_engine import (
    FetchingTask,
    FreeingTask,
    NvshmemDataEngine,
)


class BandwidthTest:
    PORT_BASE = 17777  # Base port number for communication

    def __init__(
        self,
        rank: int,
        data_engine: NvshmemDataEngine,
        min_block_size: int,
        max_block_size: int,
        num_blocks: int,
        log_file: str,
        receiver_host: str,
    ):
        self.rank = rank
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.num_blocks = num_blocks
        self.is_sender = rank == 0
        self.is_receiver = rank == 1
        self.engine = data_engine
        self.create_overheads = {}
        self.fetch_overheads = {}
        self.receiver_host = receiver_host
        self.log_dir = os.path.dirname(log_file)

        # Setup logging
        self.logger = logging.getLogger(f"bw_test_{rank}")
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

        # Setup ZMQ context and socket
        self.context = zmq.Context()
        if self.is_sender:
            self.socket = self.context.socket(zmq.DEALER)
            # Connect to receiver (rank 1) on the appropriate host
            self.socket.connect(f"tcp://{self.receiver_host}:{self.PORT_BASE}")
        else:  # receiver
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.bind(f"tcp://*:{self.PORT_BASE}")

        # Results storage
        self.results = {}

    def run_test(self):
        """Run the bandwidth test"""
        self.logger.info(
            f"Starting bandwidth test with block sizes from {self.min_block_size} to {self.max_block_size}"
        )

        num_blocks = self.num_blocks

        with self.engine:
            for block_size in range(self.min_block_size, self.max_block_size + 1):
                block_size_bytes = 2**block_size

                self.logger.info(
                    f"Testing block size: {block_size}, {num_blocks} blocks, {block_size_bytes} bytes per block"
                )

                self.create_overheads[block_size] = []
                self.fetch_overheads[block_size] = []
                if self.is_sender:
                    duration = self._send_blocks(block_size, num_blocks)
                    self.results[block_size] = duration
                    self.logger.info(f"Block size {block_size}: {duration:.6f} seconds")
                else:  # receiver
                    self._receive_blocks(block_size, num_blocks)

                # Synchronize between processes
                MPI.COMM_WORLD.Barrier()
                # Exchange fetch overhead stats so sender can print them
                if self.is_receiver:
                    mean_fetch_overhead_seconds = (
                        float(np.mean(self.fetch_overheads[block_size]))
                        if len(self.fetch_overheads[block_size]) > 0
                        else 0.0
                    )
                    MPI.COMM_WORLD.send(
                        mean_fetch_overhead_seconds, dest=0, tag=block_size
                    )
                elif self.is_sender:
                    received_mean_fetch_overhead_seconds = MPI.COMM_WORLD.recv(
                        source=1, tag=block_size
                    )
                    # Store as a single-element list to keep the existing printing logic (np.mean over list)
                    self.fetch_overheads[block_size] = [
                        received_mean_fetch_overhead_seconds
                    ]

        del self.engine
        gc.collect()

        # Print final results
        if self.is_sender:
            self._print_results()
            # Save overhead data to CSV files (only on rank 0)
            self._save_overhead_data_to_csv(self.log_dir)

    def _send_blocks(self, block_size: int, num_blocks: int) -> float:
        """Send blocks for a given block size and measure duration"""
        block_size_bytes = 2**block_size

        self.logger.info(
            f"Sending {num_blocks} blocks of size {block_size_bytes} bytes"
        )

        tensors = []
        for i in range(num_blocks):
            start_time = time.time()
            tensor = self.engine.create_tensor([block_size_bytes], dtype=torch.int8)
            end_time = time.time()
            self.create_overheads[block_size].append(end_time - start_time)
            tensors.append(tensor)

        # Warmup
        for i in range(num_blocks):
            message = {
                "type": "tensor_data",
                "id": f"{i}",
                "tensor_ptr": tensors[i].data_ptr(),
                "tensor_size": [block_size_bytes],
            }
            self.socket.send_json(message)

        for i in range(num_blocks):
            response = self.socket.recv_json()
            assert isinstance(response, dict)
            if response["status"] != "received":
                raise RuntimeError(f"Failed to send block {i} during warmup")

        start_time = time.time()

        for i in range(num_blocks):
            message = {
                "type": "tensor_data",
                "id": f"{i}",
                "tensor_ptr": tensors[i].data_ptr(),
                "tensor_size": [block_size_bytes],
            }
            self.socket.send_json(message)

        for i in range(num_blocks):
            response = self.socket.recv_json()
            assert isinstance(response, dict)
            if response["status"] != "received":
                raise RuntimeError(f"Failed to send block {i}")

        end_time = time.time()
        duration = end_time - start_time

        for i, tensor in enumerate(tensors):
            self.engine.submit_free_task(FreeingTask(tensor=tensor))

        return duration

    def _receive_blocks(self, block_size: int, num_blocks: int):
        """Receive blocks for a given block size"""
        block_size_bytes = 2**block_size

        self.logger.info(
            f"Receiving {num_blocks} blocks of size {block_size_bytes} bytes"
        )

        tensors = []

        # Warmup
        for _ in range(num_blocks):
            message = self.socket.recv_json()

            assert isinstance(message, dict)

            if message["type"] != "tensor_data":
                raise RuntimeError(f"Unexpected message type: {message['type']}")

            tensor_size = message["tensor_size"]
            fetching_task = FetchingTask(
                id=message["id"],
                remote_address=message["tensor_ptr"],
                size=tensor_size,
                dtype=torch.int8,
                remote_nvshmem_pe=0,
            )
            self.engine.submit_fetch_task(fetching_task)
            tensor = self.engine.get(message["id"])
            tensors.append(tensor)

            response = {"status": "received"}
            self.socket.send_json(response)

        for _ in range(num_blocks):
            message = self.socket.recv_json()

            assert isinstance(message, dict)

            if message["type"] != "tensor_data":
                raise RuntimeError(f"Unexpected message type: {message['type']}")

            tensor_size = message["tensor_size"]
            fetching_task = FetchingTask(
                id=message["id"],
                remote_address=message["tensor_ptr"],
                size=tensor_size,
                dtype=torch.int8,
                remote_nvshmem_pe=0,
            )
            start_time = time.time()
            self.engine.submit_fetch_task(fetching_task)
            tensor = self.engine.get(message["id"])
            end_time = time.time()
            self.fetch_overheads[block_size].append(end_time - start_time)
            tensors.append(tensor)

            response = {"status": "received"}
            self.socket.send_json(response)

        for tensor in tensors:
            self.engine.submit_free_task(FreeingTask(tensor=tensor))

    def _print_results(self):
        """Print the test results"""
        print("\n" + "=" * 60)
        print("BANDWIDTH TEST RESULTS")
        print("=" * 60)
        print(
            f"{'Block Size':<12} {'Duration (s)':<15} {'Blocks':<10} {'Total Data (MB)':<15} {'Bandwidth (GB/s)':<15} {'Create Overhead (us)':<15} {'Fetch Overhead (us)':<15}"
        )
        print("-" * 60)

        for block_size in sorted(self.results.keys()):
            duration = self.results[block_size]
            num_blocks = len(self.create_overheads[block_size])
            total_data_mb = (2**block_size * num_blocks) / (1024 * 1024)

            print(
                f"{block_size:<12} {duration:<15.6f} {num_blocks:<10} {total_data_mb:<15.2f} {total_data_mb / duration / 1024:<15.2f} {np.mean(self.create_overheads[block_size]) * 1e6:<15.6f} {np.mean(self.fetch_overheads[block_size]) * 1e6:<15.6f}"
            )

        print("=" * 60)

    def _save_overhead_data_to_csv(self, log_dir: str):
        """Save create and fetch overhead data to unified CSV file (only on rank 0)"""
        # Save unified overhead data with separate columns
        overhead_csv_path = os.path.join(log_dir, "overhead_data.csv")
        with open(overhead_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "block_size",
                    "block_size_bytes",
                    "create_overhead_microseconds",
                    "fetch_overhead_microseconds",
                ]
            )

            # Get all block sizes from both create and fetch overheads
            all_block_sizes = sorted(
                set(self.create_overheads.keys()) | set(self.fetch_overheads.keys())
            )

            for block_size in all_block_sizes:
                block_size_bytes = 2**block_size

                # Calculate average create overhead for this block size
                create_avg = (
                    np.mean(self.create_overheads[block_size]) * 1e6
                    if block_size in self.create_overheads
                    else 0.0
                )

                # Calculate average fetch overhead for this block size
                fetch_avg = (
                    np.mean(self.fetch_overheads[block_size]) * 1e6
                    if block_size in self.fetch_overheads
                    else 0.0
                )

                writer.writerow([block_size, block_size_bytes, create_avg, fetch_avg])

        self.logger.info(f"Unified overhead data saved to {overhead_csv_path}")

        # Create summary CSV with aggregated data
        self._save_summary_csv(log_dir)

    def _save_summary_csv(self, log_dir: str):
        """Save summary CSV with aggregated overhead statistics"""
        summary_csv_path = os.path.join(log_dir, "overhead_summary.csv")
        with open(summary_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "block_size",
                    "block_size_bytes",
                    "create_mean_us",
                    "create_std_us",
                    "create_min_us",
                    "create_max_us",
                    "fetch_mean_us",
                    "fetch_std_us",
                    "fetch_min_us",
                    "fetch_max_us",
                    "num_measurements",
                ]
            )

            for block_size in sorted(self.create_overheads.keys()):
                block_size_bytes = 2**block_size

                # Create overhead statistics
                create_overheads = np.array(self.create_overheads[block_size])
                create_mean = np.mean(create_overheads) * 1e6
                create_std = np.std(create_overheads) * 1e6
                create_min = np.min(create_overheads) * 1e6
                create_max = np.max(create_overheads) * 1e6

                # Fetch overhead statistics
                fetch_overheads = np.array(self.fetch_overheads[block_size])
                fetch_mean = np.mean(fetch_overheads) * 1e6
                fetch_std = np.std(fetch_overheads) * 1e6
                fetch_min = np.min(fetch_overheads) * 1e6
                fetch_max = np.max(fetch_overheads) * 1e6

                num_measurements = len(create_overheads)

                writer.writerow(
                    [
                        block_size,
                        block_size_bytes,
                        create_mean,
                        create_std,
                        create_min,
                        create_max,
                        fetch_mean,
                        fetch_std,
                        fetch_min,
                        fetch_max,
                        num_measurements,
                    ]
                )

        self.logger.info(f"Summary overhead data saved to {summary_csv_path}")

    def cleanup(self):
        """Clean up resources"""
        self.socket.close()
        self.context.term()
        self.logger.info("Bandwidth test cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Bandwidth test for distributed tensor operations"
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=10,
        help="Minimum block size (2^N bytes) (default: 10, 1KB)",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        # default=34, help="Maximum block size (2^N bytes) (default: 34, 16GB)")
        default=28,
        help="Maximum block size (2^N bytes) (default: 28, 256MB)",
    )
    parser.add_argument(
        "--num-blocks", type=int, default=128, help="Number of blocks (default: 128)"
    )
    parser.add_argument(
        "--log-dir", type=str, default=".", help="Directory to store log files"
    )
    args = parser.parse_args()

    # MPI setup
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    local_comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
    print(f"Worker {rank}/{size} (local rank {local_rank}/{local_size}) starting...")

    if size != 2:
        if rank == 0:
            print("Error: This test requires exactly 2 MPI processes")
            print(f"Current number of processes: {size}")
        sys.exit(1)

    processor_names = MPI.COMM_WORLD.allgather(MPI.Get_processor_name())
    receiver_host = processor_names[1]
    if rank == 0:
        print(f"MPI rank hosts: {processor_names}")

    # Create log file path
    log_file = os.path.join(args.log_dir, f"bw_test_{rank}.log")

    print(f"Bandwidth test process {rank}/{size} starting...")
    print(f"Block size range: 2^{args.min_block_size} to 2^{args.max_block_size} bytes")
    print(f"Number of blocks: {args.num_blocks}")

    max_block_size_bytes = 2**args.max_block_size
    num_blocks = args.num_blocks

    data_engine = NvshmemDataEngine(
        arena_size=max_block_size_bytes * num_blocks * 2,  # * 2 for warmup
        # page_size=max_block_size_bytes,
        # num_pages=num_blocks * 2, # * 2 for warmup
        # soa_buffer_size=0,
        # soa_threshold=0,
        device_id=local_rank,
        worker_id=rank,
    )

    # Create and run bandwidth test
    bw_test = BandwidthTest(
        rank=rank,
        data_engine=data_engine,
        min_block_size=args.min_block_size,
        max_block_size=args.max_block_size,
        num_blocks=args.num_blocks,
        log_file=log_file,
        receiver_host=receiver_host,
    )

    try:
        bw_test.run_test()
    finally:
        bw_test.cleanup()

    print(f"Bandwidth test process {rank} finished")


if __name__ == "__main__":
    main()
