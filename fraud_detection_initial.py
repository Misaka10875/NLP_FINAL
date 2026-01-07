import pandas as pd
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import os
from tqdm import tqdm
import sys

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

class ModelTester:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.client = OpenAI(api_key="EMPTY", base_url=config['base_url'])
        
        self.correct = 0
        self.wrong = 0
        # 诈骗判为非诈骗
        self.fraud_as_nonfraud = 0
        # 非诈骗判为诈骗
        self.nonfraud_as_fraud = 0
        self.all_results = []
        self.error_details = []
        # 诈骗判为非诈骗的详细信息
        self.fraud_as_nonfraud_details = []
        
        self.start_time = None
        self.end_time = None
        self.lock = Lock()
        
        self.pbar = None
    
    def call_model(self, content):
        prompt = f"以下对话中，right被left诈骗了吗？请只输出是或不是\n{content}"
        
        try:
            if self.model_name == 'llama':
                response = self.client.chat.completions.create(
                    model=self.config['model_name'],
                    messages=[
                        {"role": "system", "content": "You are an AI assistant. You should output in Chinese."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.config['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512
                )
            
            answer = response.choices[0].message.content.strip()
            
            return answer
        except Exception as e:
            return f"ERROR: {e}"
    
    def parse_answer(self, answer):
        answer_lower = answer.lower().strip()
        if '是' in answer_lower and '不是' not in answer_lower:
            return True
        elif '不是' in answer_lower:
            return False
        else:
            # 默认判断为非诈骗
            return False
    
    def process_row(self, row_data):
        idx, row = row_data
        content = row['specific_dialogue_content']
        is_fraud_str = str(row['is_fraud']).strip().lower()
        is_fraud = is_fraud_str == 'true'
        call_type = row['call_type']
        fraud_type = row['fraud_type']
        
        # 调用模型
        answer = self.call_model(content)
        pred_is_fraud = self.parse_answer(answer)
        
        # 判断正确性
        is_correct = (pred_is_fraud == is_fraud)
        
        with self.lock:
            # 记录所有结果
            result_row = row.copy()
            result_row['model_output'] = answer
            result_row['predicted'] = '是' if pred_is_fraud else '不是'
            result_row['actual'] = '是' if is_fraud else '不是'
            result_row['is_correct'] = is_correct
            self.all_results.append(result_row)
            
            if is_correct:
                self.correct += 1
            else:
                self.wrong += 1
                # 同时记录到错误详情
                self.error_details.append(result_row)
                
                # 统计错误类型
                if is_fraud and not pred_is_fraud:
                    self.fraud_as_nonfraud += 1
                    self.fraud_as_nonfraud_details.append({
                        'call_type': call_type,
                        'fraud_type': fraud_type
                    })
                elif not is_fraud and pred_is_fraud:
                    self.nonfraud_as_fraud += 1
            
            # 更新进度条
            if self.pbar:
                self.pbar.update(1)
                self.pbar.set_postfix({
                    '正确': self.correct,
                    '错误': self.wrong
                })
    
    def test_dataset(self, data):
        print(f"\n开始测试模型: {self.model_name}")
        self.start_time = time.time()
        
        total = len(data)
        self.pbar = tqdm(total=total, desc=f"{self.model_name}", 
                         position=list(MODELS_CONFIG.keys()).index(self.model_name),
                         leave=True)
        
        # 顺序处理每一行
        for idx, row in data.iterrows():
            self.process_row((idx, row))
        
        self.end_time = time.time()
        self.pbar.close()
    
    def generate_report(self, total_fraud, total_nonfraud):
        total = self.correct + self.wrong
        runtime = self.end_time - self.start_time
        speed = total / runtime if runtime > 0 else 0
        
        report = {
            'model': self.model_name,
            'runtime_seconds': round(runtime, 2),
            'speed_dialogues_per_sec': round(speed, 2),
            'total_correct': self.correct,
            'total_correct_percent': round(self.correct / total * 100, 2) if total > 0 else 0,
            'total_wrong': self.wrong,
            'total_wrong_percent': round(self.wrong / total * 100, 2) if total > 0 else 0,
            'dataset_total_fraud': total_fraud,
            'dataset_total_nonfraud': total_nonfraud,
            'nonfraud_as_fraud': self.nonfraud_as_fraud,
            'nonfraud_as_fraud_percent': round(self.nonfraud_as_fraud / total_nonfraud * 100, 2) if total_nonfraud > 0 else 0,
            'fraud_as_nonfraud': self.fraud_as_nonfraud,
            'fraud_as_nonfraud_percent': round(self.fraud_as_nonfraud / total_fraud * 100, 2) if total_fraud > 0 else 0,
        }
        
        # 统计诈骗判为非诈骗的各类型占比
        if self.fraud_as_nonfraud_details:
            call_type_counts = {}
            fraud_type_counts = {}
            
            for detail in self.fraud_as_nonfraud_details:
                ct = detail['call_type']
                ft = detail['fraud_type']
                call_type_counts[ct] = call_type_counts.get(ct, 0) + 1
                fraud_type_counts[ft] = fraud_type_counts.get(ft, 0) + 1
            
            report['fraud_as_nonfraud_by_call_type'] = {
                k: f"{v} ({round(v/self.fraud_as_nonfraud*100, 2)}%)" 
                for k, v in call_type_counts.items()
            }
            report['fraud_as_nonfraud_by_fraud_type'] = {
                k: f"{v} ({round(v/self.fraud_as_nonfraud*100, 2)}%)" 
                for k, v in fraud_type_counts.items()
            }
        
        return report
    
    def save_results(self, output_dir):
        # 保存所有结果
        if self.all_results:
            results_df = pd.DataFrame(self.all_results)
            results_path = os.path.join(output_dir, f'{self.model_name}_all_results.csv')
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            print(f"{self.model_name}: 所有结果已保存至 {results_path}")
        
        # 保存错误项（向后兼容）
        if self.error_details:
            error_df = pd.DataFrame(self.error_details)
            error_path = os.path.join(output_dir, f'{self.model_name}_errors.csv')
            error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
            print(f"{self.model_name}: 错误项已保存至 {error_path}")

def load_data():
    print("正在加载数据集...")
    df1 = pd.read_csv('test1.csv')
    df2 = pd.read_csv('test2.csv')
    data = pd.concat([df1, df2], ignore_index=True)
    
    # 统计诈骗和非诈骗数量
    data['is_fraud'] = data['is_fraud'].astype(str).str.strip().str.lower()
    total_fraud = (data['is_fraud'] == 'true').sum()
    total_nonfraud = (data['is_fraud'] != 'true').sum()
    
    print(f"数据集加载完成: 总计 {len(data)} 条对话")
    print(f"诈骗对话: {total_fraud}, 非诈骗对话: {total_nonfraud}")
    
    return data, total_fraud, total_nonfraud

def main():
    # 加载数据
    data, total_fraud, total_nonfraud = load_data()
    
    # 创建输出目录
    output_dir = 'initial_detection_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建测试器
    testers = {name: ModelTester(name, config) 
               for name, config in MODELS_CONFIG.items()}
    
    # 并行测试所有模型
    print("\n开始并行测试所有模型...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(tester.test_dataset, data): name 
                   for name, tester in testers.items()}
        
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\n{model_name} 测试出错: {e}")
    
    # 生成报告
    print("\n\n生成测试报告...")
    reports = []
    for name, tester in testers.items():
        report = tester.generate_report(total_fraud, total_nonfraud)
        reports.append(report)
        tester.save_results(output_dir)
        
        # 打印报告
        print(f"\n{'='*60}")
        print(f"模型: {name}")
        print(f"运行时间: {report['runtime_seconds']} 秒")
        print(f"处理速度: {report['speed_dialogues_per_sec']} 对话/秒")
        print(f"总正确: {report['total_correct']} ({report['total_correct_percent']}%)")
        print(f"总错误: {report['total_wrong']} ({report['total_wrong_percent']}%)")
        print(f"非诈骗判为诈骗: {report['nonfraud_as_fraud']} ({report['nonfraud_as_fraud_percent']}%)")
        print(f"诈骗判为非诈骗: {report['fraud_as_nonfraud']} ({report['fraud_as_nonfraud_percent']}%)")
        
        if 'fraud_as_nonfraud_by_call_type' in report:
            print(f"\n诈骗判为非诈骗按对话类型统计:")
            for ct, count in report['fraud_as_nonfraud_by_call_type'].items():
                print(f"  {ct}: {count}")
        
        if 'fraud_as_nonfraud_by_fraud_type' in report:
            print(f"\n诈骗判为非诈骗按诈骗类型统计:")
            for ft, count in report['fraud_as_nonfraud_by_fraud_type'].items():
                print(f"  {ft}: {count}")
    
    # 保存报告
    report_df = pd.DataFrame(reports)
    report_path = os.path.join(output_dir, 'detection_report.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n\n完整报告已保存至: {report_path}")

if __name__ == "__main__":
    main()


