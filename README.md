# paddlenlp_gpu_ops
paddlenlp_gpu_ops 是一个基于 PaddleNLP 的 GPU 算子库，提供了一些常用的 NLP 算子，包含 cuda 以及 triton 的实现。

# 安装

## 编译cuda算子
```bash
cd csrc
rm -rf build dist *.egg-info
python setup.py build
```

## 编译wheel包
```bash
python setup.py bdist_wheel
```

## 安装wheel包
```bash
pip install dist/*.whl
```

# Test
```bash
pytest -v tests/cuda
pytest -v tests/triton
```
