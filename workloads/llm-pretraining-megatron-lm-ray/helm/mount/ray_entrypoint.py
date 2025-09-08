import argparse
import os
import socket
import sys
import time
from contextlib import closing
from dataclasses import dataclass

import ray
import torch


@dataclass(frozen=True)
class SpecialFiles:
    """Special files that are used in the training process to drive signalling between containers"""

    done_training: str
    done_uploading: str


special_files = SpecialFiles(
    # Signal from the main training container to the sidecar that training is done
    # and all checkpoints are saved to local file system
    done_training="/local_resources/done_training",
    # Signal from the sidecar to the main training container that it can exit
    done_uploading="/local_resources/done_uploading",
)


# Helper to find a free port
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@ray.remote
class MegatronWorker:
    def __init__(self, global_rank, world_size, local_rank, master_addr, master_port, megatron_cmd_args):
        self.global_rank = global_rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port
        self.node_addr = ray.util.get_node_ip_address()
        self.megatron_cmd_args = megatron_cmd_args  # Store the command line args
        self.env_setup_done = False
        print(
            f"[Actor Init Rank {self.global_rank}] Created. local_rank={local_rank}, world_size={world_size}, master={master_addr}:{master_port}, node_addr={self.node_addr}"
        )

    def setup_environment(self):
        """
        Sets up the environment (device, env vars) required for Megatron initialization which will happen inside the pretrain function.
        """
        # Synchronize GPU Visibility Environment Variables
        hip_visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
        print(
            f"[Rank {self.global_rank}] Initial Visibility Vars: "
            f"ROCR={os.environ.get('ROCR_VISIBLE_DEVICES')}, "
            f"CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"HIP={hip_visible_devices}"
        )

        if hip_visible_devices is not None and hip_visible_devices.strip() != "":
            print(
                f"[Rank {self.global_rank}] Synchronizing ROCR/CUDA_VISIBLE_DEVICES based on HIP_VISIBLE_DEVICES='{hip_visible_devices}'"
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
            os.environ["ROCR_VISIBLE_DEVICES"] = hip_visible_devices
        else:
            print(
                f"[Rank {self.global_rank}] Warning: HIP_VISIBLE_DEVICES not set or empty. Setting visibility based on local_rank={self.local_rank}."
            )
            visible_device_str = str(self.local_rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device_str
            os.environ["ROCR_VISIBLE_DEVICES"] = visible_device_str
            os.environ["HIP_VISIBLE_DEVICES"] = visible_device_str

        print(
            f"[Rank {self.global_rank}] Synced Visibility Vars: "
            f"CUDA={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"ROCR={os.environ.get('ROCR_VISIBLE_DEVICES')}, "
            f"HIP={os.environ.get('HIP_VISIBLE_DEVICES')}"
        )
        try:
            # Set CUDA/HIP device
            if torch.cuda.is_available():
                if torch.cuda.device_count() < 1:
                    raise RuntimeError(f"[Rank {self.global_rank}] No GPUs visible to Ray actor!")
                try:
                    torch.cuda.set_device(self.local_rank)
                    print(
                        f"[Rank {self.global_rank}] Set HIP/CUDA device via local_rank to: {torch.cuda.current_device()} (intended local rank {self.local_rank})"
                    )
                except Exception as e_local:
                    print(
                        f"[Rank {self.global_rank}] Warning: Failed to set device via local_rank ({e_local}). Falling back to device 0."
                    )
                    try:
                        torch.cuda.set_device(0)  # Fallback assuming Ray isolates one GPU as device 0
                        print(
                            f"[Rank {self.global_rank}] Set HIP/CUDA device via index 0 to: {torch.cuda.current_device()}"
                        )
                    except Exception as e_zero:
                        raise RuntimeError(f"[Rank {self.global_rank}] Failed to set HIP/CUDA device 0: {e_zero}")
            else:
                print(f"[Rank {self.global_rank}] Warning: CUDA/HIP not available.")

            # Set torch.distributed variables these will be read by initialize_megatron when called inside pretrain
            os.environ["RANK"] = str(self.global_rank)
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = str(self.master_port)
            os.environ["LOCAL_RANK"] = str(self.local_rank)

            print(f"[Rank {self.global_rank}] Environment variables set for pretrain:")
            print(f"  RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")
            print(f"  MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
            print(f"  LOCAL_RANK={os.environ['LOCAL_RANK']}")

            print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"  ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
            print(f"  HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")

            self.env_setup_done = True
            print(f"[Rank {self.global_rank}] Environment setup complete.")
            return True

        except Exception as e:
            print(f"[Rank {self.global_rank}] !!!!! ERROR during setup_environment !!!!!")
            import traceback

            traceback.print_exc()
            self.env_setup_done = False
            return False

    def run_pretraining(self):
        """
        Modifies sys.argv and runs the main Megatron pretraining loop,
        which handles initialization internally via initialize_megatron.
        """
        # --- Megatron Core Imports ---
        from megatron.core.enums import ModelType
        from megatron.training import get_args, pretrain
        from megatron.training.training import destroy_global_state
        from pretrain_gpt import (
            forward_step,
            model_provider,
            train_valid_test_datasets_provider,
        )

        if not self.env_setup_done:
            print(f"[Rank {self.global_rank}] Error: Environment not set up successfully, cannot run training.")
            return False

        print(f"[Rank {self.global_rank}] Preparing to run pretrain function...")
        original_argv = sys.argv.copy()
        try:
            # Modify sys.argv before calling pretrain
            # This allows initialize_megatron (called inside pretrain) to parse the correct args
            sys.argv = ["ray_entrypoint.py"] + self.megatron_cmd_args  # Start with dummy script name
            print(f"[Rank {self.global_rank}] Temporarily setting sys.argv for pretrain: {sys.argv}")

            # Call the main pretrain function
            print(f"[Rank {self.global_rank}] Calling pretrain()...")

            train_valid_test_datasets_provider.is_distributed = True

            pretrain(
                train_valid_test_dataset_provider=train_valid_test_datasets_provider,
                model_provider=model_provider,
                model_type=ModelType.encoder_or_decoder,
                forward_step_func=forward_step,
                args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
                # extra_args_provider=None
            )

            print(f"[Rank {self.global_rank}] Pretraining function finished successfully.")
            # Restore sys.argv
            sys.argv = original_argv
            # destroy_global_state()
            return True  # Indicate success

        except Exception as e:
            print(f"[Rank {self.global_rank}] !!!!! ERROR during run_pretraining !!!!!")
            import traceback

            traceback.print_exc()  # Print full traceback
            # Ensure sys.argv is restored even on error
            sys.argv = original_argv
            # Optional cleanup on error
            # destroy_global_state()
            return False  # Indicate failure

    def get_ranks(self):
        """Returns the assigned ranks and setup status for debugging."""
        # Note: Cannot reliably get torch/mpu ranks here as init happens inside pretrain
        ranks = {
            "global_rank_assigned": self.global_rank,
            "local_rank_assigned": self.local_rank,
            "world_size_assigned": self.world_size,
            "env_setup_done": self.env_setup_done,
        }
        return ranks


# ============================================================
# Main Execution Block
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ray Launcher for Megatron-LM GPT Pretraining")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node.")

    launcher_args, megatron_script_args = parser.parse_known_args()

    total_gpus = launcher_args.num_nodes * launcher_args.gpus_per_node

    if total_gpus <= 0:
        exit(1)
    print("--- Ray Launcher Configuration ---")
    if ray.is_initialized():
        ray.shutdown()
    try:
        ray.init(ignore_reinit_error=True, log_to_driver=True)
        print(f"Ray initialized. Cluster resources: {ray.cluster_resources()}")

    except Exception as e:
        print(f"Error initializing Ray: {e}")
        exit(1)

    master_addr = ray.util.get_node_ip_address()
    master_port = find_free_port()
    print(f"Master Address for torch.distributed: {master_addr}:{master_port}")

    NUM_CPUS_PER_WORKER = os.environ.get("NUM_CPUS_PER_WORKER", 1)
    NUM_GPUS_PER_WORKER = 1

    # Create placement groups
    placement_groups = []

    world_size = total_gpus

    for node_idx in range(launcher_args.num_nodes):
        bundles = []
        if node_idx == 0:
            # Master node
            bundles = [{"CPU": 1, "GPU": 1, f"node:{master_addr}": 0.001}] * launcher_args.gpus_per_node
        else:
            bundles = [{"CPU": 1, "GPU": 1}] * launcher_args.gpus_per_node

        try:
            print(f"Creating placement group {node_idx}...")
            pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")
            timeout_seconds = 300
            ray.get(pg.ready(), timeout=timeout_seconds)
            print(f"Placement group {node_idx} ready.")
            placement_groups.append(pg)
        except Exception as e:
            print(f"Error with placement group {node_idx}: {e}")
            ray.shutdown()
            exit(1)

    # --- Launch Megatron Workers ---
    print(f"Creating {world_size} MegatronWorker actors...")
    workers = []
    for i in range(world_size):
        global_rank = i
        local_rank = i % launcher_args.gpus_per_node
        pg_index = i // launcher_args.gpus_per_node
        worker = MegatronWorker.options(
            placement_group=placement_groups[pg_index],
            placement_group_bundle_index=local_rank,
            num_cpus=NUM_CPUS_PER_WORKER,
            num_gpus=NUM_GPUS_PER_WORKER,
            resources={f"node:{master_addr}": 0.001} if pg_index == 0 else None,
        ).remote(
            global_rank=global_rank,
            world_size=world_size,
            local_rank=local_rank,
            master_addr=master_addr,
            master_port=master_port,
            megatron_cmd_args=megatron_script_args,
        )
        workers.append(worker)
    print("Actors launched.")

    # --- Trigger Environment Setup on all Actors ---
    print("\n--- Triggering Environment Setup on all workers (blocking) ---")
    setup_futures = [w.setup_environment.remote() for w in workers]
    setup_results = ray.get(setup_futures)  # Wait for all setups to complete

    if not all(setup_results):
        print("\n !!!!! ERROR: One or more workers failed during environment setup. Exiting. !!!!!")
        # Cleanup
        try:
            for pg in placement_groups:
                ray.util.remove_placement_group(pg)
        except Exception as e:
            print(f"Error removing placement group: {e}")
        for w in workers:
            ray.kill(w)
        ray.shutdown()
        exit(1)

    print("\n--- All workers completed environment setup successfully ---")

    # --- Run Pretraining ---
    print("\n--- Starting Megatron Pretraining on all workers (blocking) ---")
    # This call will block until training finishes or fails on the workers.
    training_futures = [w.run_pretraining.remote() for w in workers]

    # Wait for all workers to finish training
    training_results = ray.get(training_futures)

    print("\n--- Megatron Pretraining Finished ---")
    success = all(training_results)
    if success:
        print("All workers finished pretraining successfully.")
    else:
        print("!!!!! ERROR: One or more workers failed during pretraining. !!!!!")

    print("\n--- Shutting down Ray ---")
    try:
        for pg in placement_groups:
            ray.util.remove_placement_group(pg)
    except Exception as e:
        print(f"Error removing placement group: {e}")
    for w in workers:
        ray.kill(w)
    ray.shutdown()
    print("--- Ray Shutdown Complete ---")

    if not success:
        exit(1)  # Exit with error code if training failed

    import pathlib

    # Signal to the upload container that training is done
    pathlib.Path(special_files.done_training).touch()

    done_uploading = pathlib.Path(special_files.done_uploading)
    while not done_uploading.is_file():
        print("Waiting for upload to finish...")
        time.sleep(15)
    print("Upload finished, exiting.")
    exit(0)
