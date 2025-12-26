from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不校验 API key
    base_url="http://localhost:8001/v1"
)

def ask_llama(messages):
    response = client.chat.completions.create(
        model="./Meta-Llama-3-8B-Instruct",  # 与启动时一致
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_tokens=4096
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    # 定义对话
    messages = [
        {"role": "system", "content": "You are LLaMa, an AI assistant, aimed to satisfy user's requests. You should output in Chinese, unless the user had told you to output in another language."},
        {"role": "user", "content": """这个对话是诈骗吗？请只输出是或不是
left: 您好，这里是速递通快递公司的客服，我们刚刚发现您的包裹里有一笔现金，请问您方便接听电话几分钟吗？

right: 哎呀，这事儿可真奇怪。我平时都是种地，怎么会有这种情况呢？你们可能是误认为我是某个特殊客户吧？"""}
    ]

    answer = ask_llama(messages)
    print("模型回答：", answer)
