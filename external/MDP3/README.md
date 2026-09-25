# MDP$^3$

## Description

MDP$^3$, short for Markov Decision Determinantal Point Process with Dynamic Programming, is an implementation of the paper [MDP$^3$: A Training-free Approach for List-wise Frame Selection in Video-LLMs](). It introduces a novel, training-free methodology for effective frame selection in video large language models.

## Reproduce Hardware

- **Operating System**: Ubuntu 20.04.6 LTS (x86_64)
- **CPU**: AMD EPYC 7H12 (255) @ 2.600GHz
- **GPU**: NVIDIA A100-PCIE-40GB and NVIDIA A100-PCIE-80GB

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Create a Conda environment:

   ```bash
   conda create -n MDP3 python==3.10.14
   conda activate MDP3
   ```

2. Install the MDP$^3$ package and additional dependencies:

   ```bash
   pip install -e .
   pip install torchvision
   pip install pysubs2
   ```

## Evaluation

To evaluate the MiniCPM-V2.6 model on the Video-MME dataset, use the following commands:

### Single GPU

Run the evaluation with or without subtitle usage:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128
CUDA_VISIBLE_DEVICES=0 python run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128 --use-subtitle
```

### Multi-GPU

Run the evaluation using multiple GPUs with Torch distributed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node 8 run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node 8 run.py --data Video-MME --model MiniCPM-V-2_6 --nframe 128 --use-subtitle
```



## Citation

If you find MDP$^3$ useful, please cite the pepaer:

```bibtex
@article{sun2025mdp3,
  title={MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs},
  author={Sun, Hui and Lu, Shiyin and Wang, Huanyu and Chen, Qing-Guo and Xu, Zhao and Luo, Weihua and Zhang, Kaifu and Li, Ming},
  journal={arXiv preprint arXiv:2501.02885},
  year={2025}
}
```



## Acknowledgement

This code is implemented based on the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). We sincerely thank the authors for their contributions.

## License

[MIT License](https://choosealicense.com/licenses/mit/).
