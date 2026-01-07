# 模型下载说明

本项目需要四个开源大语言模型。由于模型文件体积较大，无法直接上传到仓库，请按照以下说明下载并放置模型文件。

## 下载链接

| 模型名称 | 下载链接 | 参数量 | 建议显存 |
|---------|---------|--------|---------|
| Qwen2.5-7B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7B | 16GB |
| Qwen3-VL-8B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 8B | 18GB |
| Meta-Llama-3-8B-Instruct | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8B | 18GB |
| Yi-1.5-9B-Chat | [HuggingFace](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) | 9B | 20GB |

## 下载方法

### 方法1: 使用 HuggingFace CLI (推荐)

首先安装 HuggingFace CLI：
```bash
pip install huggingface_hub
```

然后下载模型：

```bash
# 下载 Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./Models/Qwen2_5-7B-Instruct

# 下载 Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir ./Models/Qwen3-VL-8B-Instruct

# 下载 Meta-Llama-3-8B-Instruct
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir ./Models/Meta-Llama-3-8B-Instruct

# 下载 Yi-1.5-9B-Chat
huggingface-cli download 01-ai/Yi-1.5-9B-Chat --local-dir ./Models/Yi-1_5-9B-Chat
```

注意：
- LLaMA模型需要先在HuggingFace上申请访问权限
- 使用 `--token your_hf_token` 参数提供HuggingFace访问令牌

### 方法2: 使用 Git LFS

```bash
# 安装 Git LFS
git lfs install

# 克隆模型仓库
cd Models

git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct Qwen2_5-7B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct Qwen3-VL-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/01-ai/Yi-1.5-9B-Chat Yi-1_5-9B-Chat
```

### 方法3: 手动下载

1. 访问上述HuggingFace链接
2. 点击 "Files and versions"
3. 下载所有模型文件
4. 放置到对应的目录

## 目录结构

下载完成后，目录结构应如下：

```
Models/
├── Qwen2_5-7B-Instruct/
│   ├── config.json
│   ├── model.safetensors (或多个分片文件)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── Qwen3-VL-8B-Instruct/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── Meta-Llama-3-8B-Instruct/
│   ├── config.json
│   ├── model-*.safetensors
│   ├── tokenizer.json
│   └── ...
└── Yi-1_5-9B-Chat/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 验证下载

下载完成后，可以使用以下命令验证：

```bash
ls -la Models/Qwen2_5-7B-Instruct/
ls -la Models/Qwen3-VL-8B-Instruct/
ls -la Models/Meta-Llama-3-8B-Instruct/
ls -la Models/Yi-1_5-9B-Chat/
```

确保每个目录都包含：
- `config.json`: 模型配置文件
- `*.safetensors` 或 `pytorch_model.bin`: 模型权重文件
- `tokenizer.json` 和 `tokenizer_config.json`: 分词器文件

## 存储空间要求

请确保有足够的磁盘空间：
- Qwen2.5-7B: 约 15GB
- Qwen3-VL-8B: 约 17GB
- LLaMA-3-8B: 约 16GB
- Yi-1.5-9B: 约 18GB
- 总计: 约 66GB

## 国内下载加速

如果下载速度较慢，可以使用国内镜像：
- [阿里云ModelScope](https://modelscope.cn)
- [智源研究院](https://model.baai.ac.cn)

部分模型在ModelScope上也有镜像：
```bash
# 使用 modelscope 下载 Qwen 模型
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./Models/Qwen2_5-7B-Instruct
```

## 常见问题

### 1. 下载中断
使用 `huggingface-cli` 或 `git lfs` 下载支持断点续传，可以直接重新运行命令继续下载。

### 2. 访问权限
LLaMA模型需要在HuggingFace上申请访问权限。访问 [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 页面，点击 "Request Access"，填写表单后等待审核。

### 3. 网络问题
如果无法访问HuggingFace，可以：
- 使用代理
- 使用国内镜像站
- 寻找其他下载渠道

## 模型说明

### Qwen2.5-7B-Instruct
- 开发者: 阿里云通义千问团队
- 特点: 中文能力强，指令遵循好
- 用途: 主要测试模型之一

### Qwen3-VL-8B-Instruct
- 开发者: 阿里云通义千问团队
- 特点: 支持视觉理解，但本项目仅使用文本能力
- 用途: 对比实验

### Meta-Llama-3-8B-Instruct
- 开发者: Meta
- 特点: 开源社区广泛使用
- 用途: 主要测试模型之一

### Yi-1.5-9B-Chat
- 开发者: 零一万物
- 特点: 中英文双语能力
- 用途: 主要测试模型之一

## 许可证

请遵守各模型的许可证要求：
- Qwen系列: Apache 2.0
- LLaMA-3: Llama 3 Community License
- Yi-1.5: Apache 2.0

使用模型前请仔细阅读各自的许可证文件。

