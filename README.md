# 借助Xtuner实现针对本地数据集的模型微调与部署✨
* 借助Xtuner实现针对本地数据集的模型微调与部署

  编写时参考了xtuner官网的微调部署流程，

   Xtuner官网地址：[查看这里](https://github.com/InternLM/xtuner/blob/main/README_zh-CN.md)

通过高效、功能齐全的Xtuner工具库，可以快速地对本地数据集针对模型进行训练。

本笔记的作用为记录模型训练流程，便于日后参看，以及推荐Xtuner工具的便携与高效率，如果想要深入研究还请参考Xtuner官网的论文。

* 我们需要准备的有： 符合训练的数据集、适合微调的模型、xtuner工具库。

具体流程见下。

## 🧰 安装依赖 
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
## 🖥️ 调用配置、微调

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


**安装模型，并测试模型**
```python
pip install modelscope
from modelscope import snapshot_download, AutoTokenizer, AutoModelForCausalLM
import torch

# 将模型下载到指定的数据盘路径，这里以InternLM2-7B-Chat和 /root/autodl-tmp为例
model_dir = snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-7b", cache_dir="/root/autodl-tmp")

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

# 设置模型为推理模式
model = model.eval()

# 进行对话测试
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)
print(response)
```

**微调**
```python
xtuner train ${CONFIG_NAME_OR_PATH}
```
> 如果期望微调的效率得到优化，请加上`--deepspeed`，并且可以附加上`deepspeed_zero2`这样的优化策略。
XTuner 内置了多种策略，包括 ZeRO-1、ZeRO-2、ZeRO-3。 [——引自官网](https://github.com/InternLM/xtuner/blob/main/README_zh-CN.md)


微调结束后，在本地的work_dirs文件夹内会有保存的checkpoint,选择对应配置文件和checkpoint，指定保存路径，现在我们将模型转换为huggingface格式。

**最后，将它合并到大语言模型**

```python
xtuner convert merge./{LLM路径} ./{ADAPTER路径} ./{保存路径} --max-shard-size 2GB
```

* 这个时候，我们可以：
  1.直接使用xtuner的工具与模型对话，快捷而直接。
  2.对模型进行部署，使用任意框架，例如LMDeploy。


## 📲 部署 

**安装LMDeploy**

```python
pip install lmdeploy
```

**格式转换**
```python
lmdeploy convert ｛原模型｝｛微调模型｝
```
* 在本地生成workspace文件夹，随后通过命令行对话。


**直接调用本地模型对话**
```python
from IPython.display import display
import ipywidgets as widgets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 使用本地路径
model_name_or_path = "写入你的路径"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
model.eval()

# 初始系统提示
system_prompt = '''在这里输入你对AI的系统提示词。'''

# 初始化历史记录
history = []

# 创建输入框和按钮
input_box = widgets.Textarea(placeholder='与AI对话...')
output_box = widgets.Output()
send_button = widgets.Button(description='发送')

# 定义按钮点击事件
def on_send_button_clicked(b):
    global history
    with output_box:
        user_input = input_box.value
        input_box.value = ''
        response, history = model.chat(tokenizer, user_input, meta_instruction=system_prompt, history=history)
        print('', user_input)
        print('机器人:', response)

# 绑定事件
send_button.on_click(on_send_button_clicked)

# 显示小部件
display(input_box, send_button, output_box)
```

这样，就完成了一次模型的微调与部署，通过这个流程可以获得一份完整的模型。
