from langchain_community.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)
print("OK")