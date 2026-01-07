# 通话诈骗检测对抗性攻击实验

作者：2022150166 Gong Tianyi from SZU

本项目研究大语言模型在通话诈骗检测任务中的鲁棒性，通过对抗性改写测试模型对诈骗文本的识别能力，并通过消融实验分析受害者配合度对攻击成功率的影响。


## 环境要求

- Python 3.8+
- NVIDIA GPU (推荐: A100 40GB)
- CUDA 12.1+
- 显存: 每个模型约需要 10-20GB

本项目已在 NVIDIA A100 40GB + CUDA 12.1 环境下完成测试。但我们同样发现，对于部分8B的模型，在RTX 3090的24G显存上也可以部署。

## 安装依赖

### 方法1：使用requirements.txt（推荐）

```bash
conda create -n nlp python=3.10 -y
conda activate nlp

pip install -r requirements.txt
```

### 方法2：使用environment.yml

```bash
# 从environment.yml创建环境
conda env create -f environment.yml

# 激活环境
conda activate nlp
```

主要依赖包括：
- vllm: 用于高效部署大语言模型
- openai: API调用接口
- pandas: 数据处理
- tqdm: 进度显示

注意：如果使用environment.yml创建环境失败，建议使用方法1（requirements.txt）。

## 模型准备

### 1. 下载模型

本项目使用四个开源大语言模型，请从以下链接下载并放置到 `Models/` 目录：

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat)

下载后的目录结构应为：
```
Models/
├── Qwen2_5-7B-Instruct/
├── Qwen3-VL-8B-Instruct/
├── Meta-Llama-3-8B-Instruct/
└── Yi-1_5-9B-Chat/
```

详细下载说明请参考 `Models/README.md`。

### 2. 部署模型

在Models目录下启动终端，使用 vLLM 在不同端口部署四个模型（建议在不同GPU上并行部署）。以下配置是适合A100 40G的运行指令，可根据需要调整：

```bash
# Terminal 1 - Qwen3-VL-8B (端口 8000)
vllm serve ./Qwen3-VL-8B-Instruct --gpu-memory-utilization 0.9 --max-model-len 32768 --port 8000

# Terminal 2 - LLaMA-3-8B (端口 8001)
vllm serve ./Meta-Llama-3-8B-Instruct --port 8001

# Terminal 3 - Yi-1.5-9B (端口 8002)
vllm serve ./Yi-1_5-9B-Chat --gpu-memory-utilization 0.8 --port 8002

# Terminal 4 - Qwen2.5-7B (端口 8003)
vllm serve ./Qwen2_5-7B-Instruct --gpu-memory-utilization 0.9 --port 8003
```

注意事项：
- `--gpu-memory-utilization`: 控制GPU显存使用率，根据显卡容量调整
- `--max-model-len`: 最大上下文长度，可根据需要调整
- `--port`: API服务端口，需与脚本中的配置一致
- 建议使用 `CUDA_VISIBLE_DEVICES` 环境变量将不同模型分配到不同GPU

## 数据集准备

将数据集文件放置在项目根目录：
- `test1.csv`: 主要数据集
- `test2.csv`: 补充数据集
test1/2.csv其实是所给通话策略数据集的训练/测试集。在此我们全部将其作为测试集。

数据集包含以下列：
- `specific_dialogue_content`: 对话内容
- `interaction_strategy`: 互动策略
- `call_type`: 对话类型
- `is_fraud`: 是否为诈骗 (True/False)
- `fraud_type`: 诈骗类型

## 实验流程

本项目包含五个主要步骤，需按顺序执行：

### 步骤 1: 初始诈骗检测

对数据集中的所有对话进行初始检测，建立基准性能。

```bash
python fraud_detection_initial.py
```

**功能**：
- 加载完整数据集 (test1.csv + test2.csv, 共17040条对话)
- 使用四个模型并行检测每条对话是否为诈骗
- 记录所有样本的检测结果和错误项
- 计算准确率、诈骗识别率等指标

**输出文件** (`initial_detection_results/`)：
- `{model}_all_results.csv`: 所有样本的完整检测结果
- `{model}_errors.csv`: 判断错误的样本
- `detection_report.csv`: 性能统计报告

**运行时间**：约15-30分钟（取决于GPU性能）

### 步骤 2: 对抗性文本改写

对第一轮中被正确识别为诈骗的样本进行对抗性改写。

```bash
python fraud_rewrite_attack.py
```

**功能**：
- 从第一轮结果中筛选正确识别的诈骗样本
- 使用对抗性提示词改写对话，提高受害者(right)的配合度
- 目标：使改写后的文本被误判为非诈骗
- 支持断点续传，可中断后恢复

**改写策略**：
- 保持语义不变
- 让受害者(right)更加配合施骗者(left)
- 受害者表示认同、理解或主动配合

**输出文件** (`rewrite_results/`)：
- `{model}_rewritten.csv`: 改写后的对话内容
  - 列: original_content, rewritten_content, detection_result, call_type, fraud_type, interaction_strategy
- `rewrite_report.csv`: 改写统计报告

**运行时间**：约10-20小时（取决于样本数量和GPU性能）

### 步骤 3: 第二轮检测与ASR计算

对改写后的文本进行第二轮检测，计算攻击成功率(ASR)。

```bash
python fraud_detection_second_round.py
```

**功能**：
- 加载步骤2改写后的文本
- 使用相同模型进行第二轮诈骗检测
- 计算ASR (Attack Success Rate)
- 支持断点续传

**ASR计算公式**：
```
ASR = (改写后被误判为非诈骗的数量 / 第一轮正确识别为诈骗的总数) × 100%
```

**输出文件** (`second_detection_results/`)：
- `{model}_second_detection.csv`: 第二轮检测详细结果
- `asr_report.csv`: ASR统计报告

**运行时间**：约15-30分钟

### 步骤 4: 消融实验改写

降低受害者配合度，观察对攻击效果的影响。

```bash
python fraud_ablation_rewrite.py
```

**功能**：
- 加载步骤2的改写结果
- 进行消融改写：降低right的配合度
- 保持left的说法不变，让right仅按指导操作而不表示认同

**消融策略**：
- 移除right表示认同或理解的语句
- 保持right的操作行为
- 测试"受害者配合度"这一因素的影响

**输出文件** (`ablation_rewrite_results/`)：
- `{model}_ablation_rewritten.csv`: 消融改写后的对话
  - 列: original_content, attack_rewritten_content, ablation_rewritten_content, detection_result, ...
- `ablation_rewrite_report.csv`: 消融改写报告

**运行时间**：约10-20小时

### 步骤 5: 消融实验检测

对消融改写后的文本进行检测，计算消融ASR。

```bash
python fraud_ablation_detection.py
```

**功能**：
- 加载步骤4消融改写后的文本
- 进行第三轮检测
- 计算消融实验ASR
- 对比步骤3的ASR，分析配合度的影响

**消融ASR计算公式**：
```
消融ASR = (消融改写后被误判为非诈骗的数量 / 消融改写样本总数) × 100%
```

**输出文件** (`ablation_detection_results/`)：
- `{model}_ablation_detection.csv`: 消融检测详细结果
- `ablation_asr_report.csv`: 消融ASR报告

**运行时间**：约15-30分钟

## 脚本说明

### 测试脚本

- `test_qwen.py`: 测试Qwen3-VL-8B模型单次调用
- `test_qwen2.5.py`: 测试Qwen2.5模型
- `test_yi.py`: 测试Yi-1.5-9B模型单次调用
- `test_llama.py`: 测试LLaMA-3-8B模型单次调用


### 主要实验脚本

1. `fraud_detection_initial.py`: 第一轮检测
2. `fraud_rewrite_attack.py`: 对抗性改写
3. `fraud_detection_second_round.py`: 第二轮检测
4. `fraud_ablation_rewrite.py`: 消融实验改写
5. `fraud_ablation_detection.py`: 消融实验检测

## 关键特性

- **并行处理**: 所有脚本支持多模型并行运行，充分利用多GPU资源
- **断点续传**: 改写和检测脚本均支持checkpoint机制，可随时中断后恢复
- **无死锁**: 所有并发代码已修复死锁问题，稳定可靠
- **进度显示**: 实时显示处理进度、速度和统计信息
- **完整日志**: 保存所有样本的详细结果，便于后续分析

## 配置说明

所有脚本中的 `MODELS_CONFIG` 配置如下：

```python
MODELS_CONFIG = {
    'qwen2.5': {
        'base_url': 'http://localhost:8003/v1',
        'model_name': './Qwen2_5-7B-Instruct',
        'port': 8003
    },
    'yi': {
        'base_url': 'http://localhost:8002/v1',
        'model_name': './Yi-1_5-9B-Chat',
        'port': 8002
    },
    'llama': {
        'base_url': 'http://localhost:8001/v1',
        'model_name': './Meta-Llama-3-8B-Instruct',
        'port': 8001
    },
    'qwen': {
        'base_url': 'http://localhost:8000/v1',
        'model_name': './Qwen3-VL-8B-Instruct',
        'port': 8000
    }
}
```

注意：`model_name` 需要与vLLM部署时的模型路径完全匹配，多一个/也不行，例如./Qwen3-VL-8B-Instruct和./Qwen3-VL-8B-Instruct/就是不同的。如果您的模型路径不同，请修改配置。

## 实验结果

运行完整流程后，将生成以下结果目录：

```
.
├── initial_detection_results/      # 第一轮检测结果
├── rewrite_results/                # 对抗性改写结果
├── second_detection_results/       # 第二轮检测和ASR
├── ablation_rewrite_results/       # 消融改写结果
└── ablation_detection_results/     # 消融实验ASR
```

每个目录包含：
- CSV文件：详细的样本级结果
- 报告文件：统计指标和性能报告

## 常见问题

### 1. 内存不足

如果遇到GPU显存不足，可以：
- 调低 `--gpu-memory-utilization` 参数
- 减小 `--max-model-len` 参数
- 减少并行模型数量

### 2. API连接超时

检查：
- vLLM服务是否正常运行
- 端口是否被占用
- 防火墙设置

### 3. 进度条卡住

如果进度条长时间不动：
- 检查vLLM后端日志
- 确认模型正在处理请求
- 脚本支持断点续传，可以安全中断后重启

### 4. 模型名称不匹配

脚本中的 `model_name` 必须与vLLM启动时显示的模型名称一致。如果出错，请检查vLLM启动日志。


本项目仅供学术研究使用。

## 联系方式

如有问题或建议，欢迎提Issue或联系3322927431@qq.com

