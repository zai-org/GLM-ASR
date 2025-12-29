# GLM-ASR

[Readme in English](README.md)

<div align="center">
<img src=resources/logo.svg width="20%"/>
</div>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信</a> 社区
</p>

## 模型介绍

**GLM-ASR-Nano-2512** 是一款鲁棒的开源语音识别模型，参数量为 **1.5B**。
该模型专为应对真实世界的复杂场景而设计，在多项基准测试中超越 OpenAI Whisper V3，同时保持紧凑的模型规模。

核心能力包括：

* **卓越的方言支持**
  除标准普通话和英语外，模型针对**粤语**及其他方言进行了深度优化，有效填补了方言语音识别领域的空白。

* **低音量语音鲁棒性**
  专门针对**"低语/轻声"**场景进行训练，能够捕捉并准确转录传统模型难以识别的极低音量音频。

* **SOTA 性能**
  在同类开源模型中实现**最低平均错误率 (4.10)**，在中文基准测试（Wenet Meeting、Aishell-1 等）中展现出显著优势。

## 基准测试

我们将 GLM-ASR-Nano 与主流开源和闭源模型进行了对比评测。结果表明，**GLM-ASR-Nano (1.5B)** 表现优异，尤其在复杂声学环境下优势明显。

![bench](resources/bench.png)

说明：

* Wenet Meeting 反映了包含噪声和语音重叠的真实会议场景。
* Aishell-1 是标准普通话基准测试集。

## 模型下载

| Model             | Download Links                                                                                                                                 |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| GLM-ASR-Nano-2512 | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)<br>[🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-ASR-Nano-2512) |

+ 请注意，适配`transformers`和`SGLang`后，模型权重格式发生变化，如果你的模型下载于2025年12月27日之前，请重新拉取本版本最新模型。

## 推理

我们提供了两段测试音频，分别是中文和英语版本。

### 环境依赖

```bash
pip install -r requirements.txt
sudo apt install ffmpeg
```

### 示例代码

+ transformers 5.0.0，需源代码安装，参考 requirements.txt

```python
from transformers import AutoModel, AutoProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "zai-org/GLM-ASR-Nano-2512"

processor = AutoProcessor.from_pretrained(repo_id)
model = AutoModel.from_pretrained(repo_id, dtype=torch.bfloat16, device_map=device)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "audio",
                "url": "example_zh.wav",
            },
            {"type": "text", "text": "Please transcribe this audio into text"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
inputs = inputs.to(device, dtype=torch.bfloat16)
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

+ SGLang

目前，暂未提供发行版，请使用最新版本docker

```shell
docker pull lmsysorg/sglang:dev
```

进入docker，运行

```shell
pip install git+https://github.com/huggingface/transformers # 覆盖transformers版本
python3 -m sglang.launch_server   --model-path /cloud/oss_checkpoints/zai-org/GLM-ASR-Nano-2512 --mem-fraction-static 0.8   --served-model-name glm-asr   --host 0.0.0.0   --port 8000
```

发起请求:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8000/v1"

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
response = client.chat.completions.create(
    model="glm-asr",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": "example_zh.wav"}
                },
                {
                    "type": "text",
                    "text": "Please transcribe this audio into text"
                },
            ]
        }
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content.strip())
```
+ transformers 4.51.3 (使用未更新之前的模型)

```shell
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_en.wav # 英文
python inference.py --checkpoint_dir zai-org/GLM-ASR-Nano-2512 --audio examples/example_zh.wav # 中文
```

对于上述两段示例音频，模型能够生成准确的转录结果：

```shell
be careful not to allow fabric to become too hot which can cause shrinkage or in extreme cases scorch
我还能再搞一个，就算是非常小的声音也能识别准确
```
