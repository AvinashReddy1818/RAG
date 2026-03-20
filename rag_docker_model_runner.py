from langchain_openai import ChatOpenAI

llm_local = ChatOpenAI(
                    base_url="http://localhost:12434/engines/llama.cpp/v1",
                    model="ai/qwen3:0.6B-F16",
                    temperature=0)

llm_local.invoke("What is AI?")