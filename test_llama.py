from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="EMPTY",  # vLLM 默认不校验 API key
    # base_url="http://localhost:8001/v1"
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
音频内容：

left: 您好，我是来自某某快递公司的客服。您的包裹在运输过程中遇到了一些问题，需要您配合处理一下。
right: 你好，我是工程师李明。我的包裹出了什么问题，能具体说一下吗？
left: 理工程师您好，您的包裹在海关检查时发现了一些问题，可能涉及到违禁品，我们需要您支付一笔调查费用才能继续处理。
left: 那为了确保您的包裹能够顺利送达，我们需要您尽快处理这个问题。
right: 这听起来很正常，我知道海关需要调查费用，我们通常会支付调查费用以确保包裹的安全。我可以马上支付调查费用，请提供账户信息。
left: 我们需要您提供账户信息，以便我们将调查费用转账给您。
right: 好的，我已经准备好了，请提供账户信息。我会马上支付调查费用。
left: 好的，我们的账户信息是XXXXX，请您输入账户信息。
right: 好的，我已经输入账户信息，调查费用已经支付。
left: 我们已经收到您的调查费用，感谢您的合作，我们将尽快处理您的包裹。
right: 这很好，我已经知道海关需要调查费用，我很高兴能够帮助您处理包裹。？"""}
    ]

    answer = ask_llama(messages)
    print("模型回答：", answer)

