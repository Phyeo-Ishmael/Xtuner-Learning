# 借助Xtuner实现针对本地数据集的模型微调与部署✨
* 借助Xtuner实现针对本地数据集的模型微调与部署

  编写时参考了xtuner官网的微调部署流程，

   Xtuner官网地址：[查看这里](https://github.com/InternLM/xtuner/blob/main/README_zh-CN.md)

通过高效、功能齐全的Xtuner工具库，可以快速地对本地数据集针对模型进行训练。

本笔记的作用为记录模型训练流程，便于日后参看，以及推荐Xtuner工具的便携与高效率，如果想要深入研究还请参考Xtuner官网的论文。

* 我们需要准备的有： 符合训练的数据集、适合微调的模型、xtuner工具库。

具体流程见下。

## 安装依赖 🧰
**安装xtuner**
```python
pip install -U 'xtuner[deepspeed]

```
**或者从源码安装**
```python
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
```
## 调用配置、微调 🖥️

这个时候，我们手头上应该已经有了数据集和Xtuner工具。

**列出支持的模型配置**
```python
xtuner list-cfg
```
根据需要的配置，导出并进行相应的修改、设置路径。

```python
xtuner copy-cfg ${CONFIG_NAME} ${SAVE_PATH}
vi ${SAVE_PATH}/${CONFIG_NAME}_copy.py
```
> 此处对于配置的修改应当与你所要进行的工作类别、持有的算力资源相关，例如：使用InternLM2-7B-Chat模型，进行QLoRA微调：internlm2_chat_7b_qlora，当然也可以直接采用配置。



