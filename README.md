# 借助Xtuner实现针对本地数据集的模型微调与部署✨
* 借助Xtuner实现针对本地数据集的模型微调与部署
    Xtuner地址：[查看这里](https://github.com/InternLM/xtuner/blob/main/README_zh-CN.md)

通过高效、功能齐全的Xtuner工具库，可以快速地对本地数据集针对模型进行训练。

本笔记的作用为记录模型训练流程，便于日后参看，以及推荐Xtuner工具的便携与高效率，如果想要深入研究还请参考Xtuner官网的论文。

*我们需要准备的有： 符合训练的数据集、适合微调的模型、xtuner工具库。

## 安装依赖 🧰
**安装xtuner**
```python
pip install -U 'xtuner[deepspeed]

```
**或者从源码安装**
```python
git clone https://github.com/InternLM/xtuner.git\n
```
