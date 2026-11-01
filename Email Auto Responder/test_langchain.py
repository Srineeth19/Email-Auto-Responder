from langchain_ollama import ChatOllama

# Create the Ollama model
model = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Send a test question
response = model.invoke(
    "Explain what an email assistant does in one sentence."
)

# Display the response
print(response.content)