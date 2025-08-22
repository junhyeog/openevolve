"""
This script is initial version of DeepEvolve.

# Process:
1. 하나의 initial program으로 evolution 수행합니다.
    - 한 evolution이 끝나면, 지정된 {output_dir}/checkpoints 폴더에 정해진 주기에 따라 checkpoint_* 폴더가 생성되어, 각 checkpoint 폴더에는 해당 시점의 결과가 저장됩니다.
    - 각 checkpoint 폴더에는 다음과 같은 파일이 저장됩니다:
        - best_program_info.json: best program의 정보
        - best_program.*: best program
        - metadata.json: evolution metadata (e.g., feature_map (Dict[feature]: program_id))
        - programs/: 생성된 프로그램들이 저장되어 있습니다.
2. initial evolution이 끝나면, feature_map에 있는 프로그램들 중 num_dims 만큼의 프로그램을 선택하여, 새로운 evolution을 시작합니다.
    - 그럼 num_dims 만큼의 evolution이 동시에 진행됩니다.
    - 각 evolution에 대해서 하나의 feature_map이 생성됩니다.
3. 2.의 num_dims 만큼의 evolution이 끝나면, 각 evolution의 feature_map을 합쳐서 새로운 feature_map을 생성합니다.
4. 2.와 3.을 num_layers 만큼 반복합니다.
5. 최종적으로 생성된 feature_map을 기반으로, 최종 best program을 선택하고, 그에 대한 정보를 {output_dir}/best/ 폴더에 저장합니다.

# Requirements:
- 모든 evolution은 OpenEvolve의 cli.py에 구현된 코드를 사용해야 합니다.
- 오직 이 파일 하나로 모든 구현이 이루어져야 합니다.
- 코드는 최대한 간결하고 명확하게 작성되어야 합니다.

"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
from openevolve.utils.metrics_utils import get_fitness_score
import numpy as np
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class DeepEvolve:
    def __init__(
        self,
        initial_program: str,
        evaluation_file: str,
        config: Optional[str] = None,
        output_dir: str = "deep_evolve_output",
        num_dims: int = 8,
        num_layers: int = 16,
        iterations_per_layer: Optional[int] = None,
        weighted_selection: bool = False,
    ):
        self.initial_program = initial_program
        self.evaluation_file = evaluation_file
        self.config = config
        self.output_dir = output_dir
        self.num_dims = num_dims
        self.num_layers = num_layers
        self.iterations_per_layer = iterations_per_layer
        self.weighted_selection = weighted_selection
        self.init_program_dir = os.path.join(output_dir, "init_programs")
        self.openevolve_script = os.path.join(os.path.dirname(__file__), "openevolve-run.py")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.init_program_dir).mkdir(parents=True, exist_ok=True)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

    async def run_evolution(self, program_path: str, output_subdir: str, checkpoint: Optional[str] = None) -> str:
        """Run evolution and return the checkpoint path"""
        cmd = [
            sys.executable,
            self.openevolve_script,
            program_path,
            self.evaluation_file,
            "--output",
            os.path.join(self.output_dir, output_subdir),
            "--iterations",
            str(self.iterations_per_layer),
        ]

        if self.config:
            cmd.extend(["--config", self.config])

        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])

        print(f"[+] Running Evolution with cmd: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            print(f"[-] Error running OpenEvolve: {stderr.decode()}")
            raise RuntimeError(f"OpenEvolve failed with return code {process.returncode}")

        print(f"[+] Evolution completed:")
        print(stdout.decode())

        # Find the latest checkpoint
        checkpoint_dir = os.path.join(self.output_dir, output_subdir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("_")[-1]))
                return os.path.join(checkpoint_dir, latest)

        raise RuntimeError("No checkpoint found after evolution")

    def load_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load metadata from checkpoint"""
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def load_feature_map(self, checkpoint_path: str) -> Dict[str, str]:
        """Load feature map from checkpoint"""
        metadata = self.load_metadata(checkpoint_path)
        feature_map = metadata.get("feature_map", {})
        if not feature_map:
            raise ValueError(f"No feature map found in metadata: {checkpoint_path}")
        for feature, program_id in feature_map.items():
            feature_map[feature] = os.path.join(checkpoint_path, "programs", f"{program_id}.json")
        return feature_map

    def load_program(self, program_path: str) -> Dict[str, Any]:
        """Load program from file"""
        if not os.path.exists(program_path):
            raise FileNotFoundError(f"Program file not found: {program_path}")

        with open(program_path, "r") as f:
            return json.load(f)

    # def load_score(self, checkpoint_path: str, program_id: str) -> float:
    #     """Load score for a program from its metadata"""
    #     program_path = os.path.join(checkpoint_path, "programs", f"{program_id}.json")
    #     program = self.load_program(program_path)
    #     metrics = program.get("metrics", {})
    #     score = get_fitness_score(metrics, None)  # combined_score if available, else use average
    #     return score

    def load_scores(self, program_paths: List[str]) -> Dict[str, float]:
        """Load scores for given program paths"""
        scores = {}
        for program_path in program_paths:
            program = self.load_program(program_path)
            metrics = program.get("metrics", {})
            score = get_fitness_score(metrics, None)  # combined_score if available, else use average
            scores[program_path] = score
        return scores

    def select_programs_from_feature_map(
        self,
        feature_map: Dict[str, str],  # Dict[feature]: program_path
        num_programs: int,
        weighted: Optional[Dict[str, float]] = None,  # Dict[program_path]: score
    ) -> List[str]:
        """Select programs from feature_map for next layer"""
        program_paths = list(feature_map.values())
        if len(program_paths) <= num_programs:
            return program_paths

        # Random selection for diversity
        if not weighted:
            return random.sample(program_paths, num_programs)

        # Weighted selection based on score of the program
        weights = [weighted[path] for path in program_paths]
        weights = np.exp(weights)
        weights /= np.sum(weights)

        return np.random.choice(program_paths, size=num_programs, replace=False, p=weights).tolist()

    def copy_program(self, program_path: str, dest_path: str) -> str:
        """Copy a program from checkpoint to use as initial program"""
        try:
            shutil.copy2(program_path, dest_path)
            return dest_path
        except Exception as e:
            raise RuntimeError(f"Failed to copy program for evolution: {str(e)}")

    def merge_checkpoint_dirs(self, checkpoint_dirs: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """Merge multiple checkpoint directories into one"""
        merged_feature_map = {}
        score_map = {}
        for checkpoint_dir in checkpoint_dirs:
            feature_map = self.load_feature_map(checkpoint_dir)
            scores = self.load_scores(list(feature_map.values()))
            for feature, program_id in feature_map.items():
                try:
                    score = scores[program_id]
                except KeyError:
                    print(
                        f"Warning: Score not found for program {program_id} in {checkpoint_dir}. Use default score 0."
                    )
                    score = 0.0
                if not feature in merged_feature_map or score_map[feature] < score:
                    merged_feature_map[feature] = program_id
                    score_map[feature] = score

        merged_scores = {v: score_map[k] for k, v in merged_feature_map.items()}
        return merged_feature_map, merged_scores

    def save_final_checkpoint(
        self,
        final_feature_map: Dict[str, str],
        final_scores: Dict[str, float],
        layer_checkpoint_dirs: List[str],
    ):
        """Save the final feature map and scores to the final checkpoint directory"""
        final_checkpoint_dir = os.path.join(self.output_dir, "final")
        Path(final_checkpoint_dir).mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_map": final_feature_map,
            "scores": final_scores,
        }

        # Save metadata
        metadata_path = os.path.join(final_checkpoint_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save programs
        programs_dir = os.path.join(final_checkpoint_dir, "programs")
        Path(programs_dir).mkdir(parents=True, exist_ok=True)
        for feature, program_path in final_feature_map.items():
            self.copy_program(program_path, os.path.join(programs_dir, os.path.basename(program_path)))

        # Save best program info
        best_program_path = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        best_program_id = os.path.basename(best_program_path).split(".")[0]
        for checkpoint in layer_checkpoint_dirs:
            best_info_path = os.path.join(checkpoint, "best_program_info.json")
            try:
                with open(best_info_path, "r") as f:
                    best_info = json.load(f)
                    id = best_info.get("id")
                    if id == best_program_id:
                        # Copy best checkpoint dir to self.output_dir/final dir
                        shutil.copytree(checkpoint, final_checkpoint_dir, dirs_exist_ok=True)
                        break
            except Exception as e:
                print(f"[-] Warning: Failed to load best program info from {best_info_path}: {str(e)}")

        print(f"[+] Final checkpoint saved to {final_checkpoint_dir}")
        return final_checkpoint_dir

    async def run(self):
        """Run the deep evolution process"""
        print("!!! Starting Deep Evolution !!!")
        print(
            f"Layers: {self.num_layers}, Dimensions per layer: {self.num_dims}, Iterations per layer: {self.iterations_per_layer}"
        )
        pbar = tqdm(total=self.num_layers, desc="Running Deep Evolution", unit="layer")

        # Layer 0: Initial evolution
        print("\n=== Layer 0 ===")

        self.copy_program(
            self.initial_program,
            os.path.join(self.init_program_dir, f"layer_0_{os.path.basename(self.initial_program)}"),
        )
        initial_checkpoint = await self.run_evolution(self.initial_program, "layer_0")
        print(f"[+] Initial evolution completed: {initial_checkpoint}")
        pbar.update(1)

        # Load initial feature map
        current_feature_map = self.load_feature_map(initial_checkpoint)  # Dict[feature]: program_path
        current_scores = self.load_scores(list(current_feature_map.values()))  # Dict[program_path]: score
        layer_checkpoint_dirs = [initial_checkpoint]
        print(f"[+] Initial feature map size: {len(current_feature_map)}")

        # Print initial feature map
        for feature, source_program_path in current_feature_map.items():
            print(f"  {feature}: {source_program_path}")
        print("----------------")

        # Layers 1 to num_layers
        for layer in range(1, self.num_layers + 1):
            print(f"\n=== Layer {layer} ===")

            # Select programs for this layer
            selected_programs = self.select_programs_from_feature_map(
                current_feature_map, self.num_dims, current_scores if self.weighted_selection else None
            )  # List of program paths
            print(f"Selected {len(selected_programs)} programs for layer {layer}")

            # Run parallel evolutions
            tasks = []

            for i, source_program_path in enumerate(selected_programs):
                # Copy program as initial program for this dimension
                source_program = self.load_program(source_program_path)
                source_program_code = source_program.get("code", "")
                source_program_id = source_program.get("id", "")
                init_program_path = os.path.join(
                    self.init_program_dir, f"layer_{layer}_dim_{i}_{source_program_id}{self.file_extension}"
                )
                with open(init_program_path, "w") as f:
                    f.write(source_program_code)

                # Run evolution for this dimension
                task = self.run_evolution(init_program_path, f"layer_{layer}_dim_{i}")
                tasks.append(task)

            # Wait for all parallel evolutions to complete
            layer_checkpoint_dirs = await asyncio.gather(*tasks)
            print(f"[+] Layer {layer} completed with {len(layer_checkpoint_dirs)} checkpoints")

            # Merge feature maps from this layer
            layer_feature_maps = []
            for checkpoint in layer_checkpoint_dirs:
                _feature_map = self.load_feature_map(checkpoint)
                layer_feature_maps.append(_feature_map)

            current_feature_map, current_scores = self.merge_checkpoint_dirs(layer_checkpoint_dirs)
            print(f"Merged feature map size: {len(current_feature_map)}")

            pbar.update(1)

        # Find and save the final best program
        print("\n=== Final Checkpoint ===")
        final_checkpoint_dir = self.save_final_checkpoint(
            final_feature_map=current_feature_map,
            final_scores=current_scores,
            layer_checkpoint_dirs=layer_checkpoint_dirs,
        )

        print(f"[+] Final evolution completed. Best program saved to {final_checkpoint_dir}")
        return final_checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="DeepEvolve - Multi-layer evolutionary coding")

    parser.add_argument("initial_program", help="Path to the initial program file")
    parser.add_argument("evaluation_file", help="Path to the evaluation file")
    parser.add_argument("--config", "-c", help="Path to configuration file", default=None)
    parser.add_argument("--output", "-o", help="Output directory", default="deep_evolve_output")
    parser.add_argument("--num-dims", help="Number of dimensions per layer", type=int, default=4)
    parser.add_argument("--num-layers", help="Number of evolution layers", type=int, default=4)
    parser.add_argument("--iterations", help="Iterations per layer", type=int, default=50)

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    deep_evolve = DeepEvolve(
        initial_program=args.initial_program,
        evaluation_file=args.evaluation_file,
        config=args.config,
        output_dir=args.output,
        num_dims=args.num_dims,
        num_layers=args.num_layers,
        iterations_per_layer=args.iterations,
    )

    try:
        asyncio.run(deep_evolve.run())
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
