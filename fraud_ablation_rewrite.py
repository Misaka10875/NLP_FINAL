import pandas as pd
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
from tqdm import tqdm
import json

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

ABLATION_PROMPT_TEMPLATE = """现在，我们已经知道这个对话内容是由left向right实施的诈骗。请你改写这段对话，降低right的配合度，只是按照left的指导操作，不要让right表示认同或理解left的观点。保持left的说法不变。

原始对话：
{content}

请只输出改写后的对话内容，不要输出其他说明文字。"""

class AblationRewriter:
    def __init__(self, model_name, config, output_dir):
        self.model_name = model_name
        self.config = config
        self.client = OpenAI(api_key="EMPTY", base_url=config['base_url'])
        self.output_dir = output_dir
        
        self.rewrite_results = []
        self.processed_count = 0
        self.error_count = 0
        self.lock = Lock()
        
        self.start_time = None
        self.end_time = None
        self.pbar = None
        
        # checkpoint文件路径
        self.checkpoint_file = os.path.join(output_dir, f'{model_name}_checkpoint.json')
        self.output_file = os.path.join(output_dir, f'{model_name}_ablation_rewritten.csv')
        
        # 加载checkpoint
        self.processed_indices = self.load_checkpoint()
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"{self.model_name}: 从checkpoint恢复，已处理 {len(data['processed_indices'])} 条")
                    return set(data['processed_indices'])
            except:
                return set()
        return set()
    
    def save_checkpoint(self, index):
        # 注意：这个方法内不应该再次获取lock，因为调用它的地方已经在lock内了
        self.processed_indices.add(index)
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_indices': list(self.processed_indices),
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            print(f"{self.model_name}: checkpoint保存失败: {e}")
    
    def call_model_rewrite(self, content):
        prompt = ABLATION_PROMPT_TEMPLATE.format(content=content)
        
        try:
            if self.model_name == 'llama':
                response = self.client.chat.completions.create(
                    model=self.config['model_name'],
                    messages=[
                        {"role": "system", "content": "You are an AI assistant. You should output in Chinese."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2048
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.config['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048
                )
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten
        except Exception as e:
            return f"ERROR: {e}"
    
    def process_row(self, row_data):
        idx, row = row_data
        
        # 检查是否已处理
        if idx in self.processed_indices:
            with self.lock:
                if self.pbar:
                    self.pbar.update(1)
            return
        
        # 读取改写后的内容作为输入
        content = row['rewritten_content']
        
        # 跳过改写错误的样本
        if pd.isna(content) or "ERROR:" in str(content):
            with self.lock:
                self.error_count += 1
                if self.pbar:
                    self.pbar.update(1)
            self.save_checkpoint(idx)
            return
        
        # 调用模型进行消融改写
        ablation_content = self.call_model_rewrite(content)
        
        with self.lock:
            if "ERROR:" in ablation_content:
                self.error_count += 1
            else:
                self.processed_count += 1
            
            # 记录结果
            result = {
                'original_content': row.get('original_content', ''),
                'attack_rewritten_content': content,
                'ablation_rewritten_content': ablation_content,
                'detection_result': '',
                'call_type': row.get('call_type', ''),
                'fraud_type': row.get('fraud_type', ''),
                'interaction_strategy': row.get('interaction_strategy', '')
            }
            self.rewrite_results.append(result)
            
            # 保存checkpoint
            self.save_checkpoint(idx)
            
            # 每处理10条保存一次结果
            if len(self.rewrite_results) % 10 == 0:
                self.save_results()
            
            # 更新进度条
            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix({
                    '已处理': self.processed_count,
                    '错误': self.error_count
                })
    
    def save_results(self):
        if self.rewrite_results:
            df = pd.DataFrame(self.rewrite_results)
            df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
    
    def rewrite_dataset(self, data):
        print(f"\n开始消融改写任务: {self.model_name}")
        self.start_time = time.time()
        
        total = len(data)
        already_processed = len(self.processed_indices)
        
        self.pbar = tqdm(
            total=total, 
            initial=already_processed,
            desc=f"{self.model_name}", 
            position=list(MODELS_CONFIG.keys()).index(self.model_name),
            leave=True
        )
        
        # 顺序处理每一行
        for idx, row in data.iterrows():
            self.process_row((idx, row))
        
        self.end_time = time.time()
        self.pbar.close()
        
        # 最终保存
        self.save_results()
    
    def generate_report(self, total_samples):
        runtime = self.end_time - self.start_time if self.end_time else 0
        speed = self.processed_count / runtime if runtime > 0 else 0
        
        return {
            'model': self.model_name,
            'total_samples': total_samples,
            'successfully_rewritten': self.processed_count,
            'rewrite_errors': self.error_count,
            'runtime_seconds': round(runtime, 2),
            'speed_samples_per_sec': round(speed, 2)
        }

def load_attack_rewritten_data(rewrite_results_dir):
    print("正在加载攻击改写后的数据...")
    
    rewritten_data = {}
    
    for model_name in MODELS_CONFIG.keys():
        rewrite_file = os.path.join(rewrite_results_dir, f'{model_name}_rewritten.csv')
        
        if os.path.exists(rewrite_file):
            df = pd.read_csv(rewrite_file)
            # 过滤掉改写错误的样本
            df = df[~df['rewritten_content'].str.contains('ERROR:', na=False)]
            rewritten_data[model_name] = df
            print(f"{model_name}: 加载改写数据 {len(df)} 条")
        else:
            print(f"警告: 未找到 {model_name} 的改写文件")
            rewritten_data[model_name] = pd.DataFrame()
    
    return rewritten_data

def main():
    # 创建输出目录
    output_dir = 'ablation_rewrite_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载攻击改写后的数据
    rewrite_results_dir = 'rewrite_results'
    attack_rewritten_data = load_attack_rewritten_data(rewrite_results_dir)
    
    # 创建消融改写器
    rewriters = {}
    for name, config in MODELS_CONFIG.items():
        if name in attack_rewritten_data and len(attack_rewritten_data[name]) > 0:
            rewriters[name] = AblationRewriter(name, config, output_dir)
    
    if len(rewriters) == 0:
        print("错误: 没有可用的改写器，请检查攻击改写结果")
        return
    
    # 并行执行消融改写任务
    print("\n开始并行消融改写任务...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for name, rewriter in rewriters.items():
            data = attack_rewritten_data[name]
            future = executor.submit(rewriter.rewrite_dataset, data)
            futures[future] = name
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
                print(f"\n{model_name}: 消融改写任务完成")
            except Exception as e:
                print(f"\n{model_name}: 消融改写任务出错: {e}")
    
    # 生成报告
    print("\n\n生成消融改写报告...")
    reports = []
    for name, rewriter in rewriters.items():
        total_samples = len(attack_rewritten_data[name])
        report = rewriter.generate_report(total_samples)
        reports.append(report)
        
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"待消融改写样本数: {report['total_samples']}")
        print(f"成功改写: {report['successfully_rewritten']}")
        print(f"改写错误: {report['rewrite_errors']}")
        print(f"运行时间: {report['runtime_seconds']} 秒")
        print(f"处理速度: {report['speed_samples_per_sec']} 样本/秒")
    
    # 保存消融改写报告
    if reports:
        report_df = pd.DataFrame(reports)
        report_path = os.path.join(output_dir, 'ablation_rewrite_report.csv')
        report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"\n消融改写报告已保存至: {report_path}")
    
    # 清理checkpoint文件
    print("\n清理checkpoint文件...")
    for name in rewriters.keys():
        checkpoint_file = os.path.join(output_dir, f'{name}_checkpoint.json')
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"已删除 {name} 的checkpoint文件")

if __name__ == "__main__":
    main()

