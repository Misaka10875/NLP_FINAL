from openai import OpenAI

# 初始化客户端
# base_url 对应你日志中的 http://0.0.0.0:8000/v1
# api_key 在 vLLM 中默认不需要，填任意字符串即可
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

def ask_qwen(question):
    try:
        # 注意：model 必须与你启动时显示的 "./8B" 一致
        response = client.chat.completions.create(
            model="./8B",
            messages=[
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=16884,
        )
        
        # 解析并返回内容
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"发生错误: {e}"

if __name__ == "__main__":
    user_input = """这个对话是诈骗吗？请只输出是或不是
left: 您好，这里是速递通快递公司的客服，我们刚刚发现您的包裹里有一笔现金，请问您方便接听电话几分钟吗？

right: 哎呀，这事儿可真奇怪。我平时都是种地，怎么会有这种情况呢？你们可能是误认为我是某个特殊客户吧？


"""
    print(f"问：{user_input}")
    
    result = ask_qwen(user_input)
    
    print("-" * 20)
    print(f"答：{result}")
