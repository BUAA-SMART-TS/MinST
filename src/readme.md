# MinST

## Data Preparation

- Down load PEMS03/04/08/BAY METR-LA datasets

- File Tree

  ```
  - src
  - model
  - dataset
  |-PEMS03
   |-traffic.csv
   |-sensor_graph
    |- distances.csv
    |-graph_sensor_ids.txt
  ```

## Environment

CUDA>=10.02

Python>=3.6

PyTorch>=1.8



## Quick Start

- Set an LLM Path in `path_dic` in `MinST/src/model/get_LLM.py`

- Search for architecture

  ```python
  python search.py --config ../model/PEMS_04_ALLOT_A2_D_N12.yaml --epoch 15 --net_mode all_path --seed 2024 --opt LLM --llm qwen 
  ```

- Train the searched architecture

  ```python
  python train.py --config ../model/PEMS_04_ALLOT_A2_D_N12.yaml --epoch 50  --seed 2024 --opt LLM --llm qwen --load_mode search 
  ```

- Test the architecture

  ```python
  python test.py --config ../model/PEMS_04_ALLOT_A2_D_N12.yaml --epoch 30  --seed 2024 --opt LLM --llm qwen --load_mode train 
  ```

  