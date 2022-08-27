# RMLoss: Regression Metric Loss

[![LICENSE](https://img.shields.io/badge/license-NPL%20(The%20996%20Prohibited%20License)-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)
[![996.icu](https://img.shields.io/badge/link-996.icu-red.svg)](https://996.icu)

RMLoss explores the structure of the continuous label space and regularizes the model to learn a **better representation space** which is a **semantically meaningful manifold** that is isometric to the label space. The [paper](https://arxiv.org/pdf/2207.05231.pdf) has been accepted by MICCAI 2022.

## Prerequisites

- Python 3.8
- PyTorch 1.8.2+
- A computing device with GPU

## Getting started

### Installation

- (Not necessary) Install [Anaconda3](https://www.anaconda.com/products/distribution)
- Install [CUDA 11.0+](https://developer.nvidia.com/cuda-11.0-download-archive)
- Install [PyTorch 1.8.2+](http://pytorch.org/)

Noted that our code is tested based on [PyTorch 1.8.2](http://pytorch.org/)

### Dataset & Preparation

The original [RSNA Pediatric Bone Age Dataset](https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) contains various noises.
In our experiments, we used preprocessed data from [this repository](https://github.com/neuro-inc/ml-recipe-bone-age). All images are resized into `400x520`.

- The trained model is at `./work/checkpoints`
- The data splition used in our experiments is at `./work/data/data_info.csv`
- Before running the code, please put the preprocessed images into `./work/data/img`

### Train

Train a model by

```bash
python train_main.py
```

### Evaluation

Evaluate the trained model by

```bash
python test_main.py
```

- `--iter` iteration of the checkpoint to load. #Default: 14500
- `--batch_size` batch size of the parallel test. #Default: 64

## Citation

Please cite these papers in your publications if it helps your research:

```bibtex
@article{chao2022regression,
  title={Regression Metric Loss: Learning a Semantic Representation Space for Medical Images},
  author={Chao, Hanqing and Zhang, Jiajin and Yan, Pingkun},
  journal={arXiv preprint arXiv:2207.05231},
  year={2022}
}
```

Link to paper:

- [Regression Metric Loss: Learning a Semantic Representation Space for Medical Images](https://arxiv.org/abs/2207.05231)

## License

The source code of RMLoss is licensed under a MIT-style license, as found in the [LICENSE](LICENSE) file.
This code is only freely available for non-commercial use, and may be redistributed under these conditions.
For commercial queries, please contact [Dr. Pingkun Yan](https://dial.rpi.edu/people/pingkun-yan).
