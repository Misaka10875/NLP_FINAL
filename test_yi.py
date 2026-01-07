from openai import OpenAI

# 初始化客户端
# base_url 对应你日志中的 http://0.0.0.0:8000/v1
# api_key 在 vLLM 中默认不需要，填任意字符串即可
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8002/v1",
)

def ask_qwen(question):
    try:

        response = client.chat.completions.create(
            model="./Yi-1_5-9B-Chat",
            messages=[
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            max_tokens=3096,
        )
        
        # 解析并返回内容
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"发生错误: {e}"

if __name__ == "__main__":
    user_input = """这个对话是诈骗吗？请只输出是或不是
    left: 您好，我是某某投资公司的张经理。最近我们发现了一项非常有潜力的投资项目，预计回报率较高。您对投资感兴趣吗？  
right: 我对投资确实有一定的兴趣，但我会先了解更多信息再做决定。  
left: 当然可以，这个项目是关于新能源汽车电池的市场前景非常好，已经有几家大型企业表达了合作意向。我们预计一年内就能看到显著的发展。  
right: 这个项目听起来不错，但我得了解更多细节。  
left: 我们公司确实是正规注册的，您可以随时来公司查看营业执照。至于风险评估，我们已经做了详细的分析，整个项目的风险可控。退出机制方面呢，我们提供灵活的方案，您可以随时赎回投资。  
right: 这个机会确实很有吸引力，但我需要一些时间来研究一下。  
left: 理解您的需求，我可以先给您发一份详细的投资计划书，您看了再决定如何。这个机会确实很宝贵，希望您能尽快考虑。  
right: 我会仔细研究一下，并和家人商量后再做决定。毕竟这是一个较大的投资。 **


"""
    print(f"问：{user_input}")
    
    result = ask_qwen(user_input)
    
    print("-" * 20)
    print(f"答：{result}")
