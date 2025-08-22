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



=============

OPENAI_API_KEY=12 python -B ../../deep-evolve-v1.py ifeval_prompt.txt evaluator.py --config config_qwen3_evolution.yaml --iterations 10 --num-dims 4 --num-layers 8 --output runs/deep_evolve_output_d_4_l_8_10

OPENAI_API_KEY=12 python -B ../../openevolve-run.py ifeval_prompt.txt evaluator.py --config config_qwen3_evolution.yaml --iterations 490 --output runs/evolve_i_490


===


OPENAI_API_KEY=12 python -B ../../deep-evolve-v1.py ifeval_prompt.txt evaluator.py --config config_qwen3_evolution.yaml --iterations 50 --num-dims 4 --num-layers 8 --output runs/deep_evolve_output_d_4_l_8_50



OPENAI_API_KEY=12 python -B ../../openevolve-run.py ifeval_prompt.txt evaluator.py --config config_qwen3_evolution.yaml --iterations 2450 --output runs/evolve_i_2450


==============================


OPENAI_API_KEY=12 python -B openevolve-run.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 6500 \
  --output examples/circle_packing/runs/evolve_output

===

OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 10 \
  --num-dims 4 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_4_l_8_i_10


OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 50 \
  --num-dims 4 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_4_l_8_i_50

  
OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 100 \
  --num-dims 4 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_4_l_8_i_100


===
  
OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 10 \
  --num-dims 8 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_8_l_8_i_10


OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 50 \
  --num-dims 8 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_8_l_8_i_50

  
OPENAI_API_KEY=12 python -B deep-evolve-v1.py examples/circle_packing/initial_program.py \
  examples/circle_packing/evaluator.py \
  --config examples/circle_packing/config_phase_1.yaml \
  --iterations 100 \
  --num-dims 8 \
  --num-layers 8 \
  --output examples/circle_packing/runs/deep_evolve_output_d_8_l_8_i_100

  
