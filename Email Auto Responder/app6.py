from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import re


# ============================================================
# 1. OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# 2. ORDER STATUS TOOL
# ============================================================

@tool
def get_order_status(order_id: str):
    """
    Check the status of a customer's order using the order ID.
    """

    order_database = {
        "ORD123": "Shipped",
        "ORD456": "Processing",
        "ORD789": "Delivered",
        "ORD101": "Cancelled"
    }

    return order_database.get(
        order_id.upper(),
        "Order ID not found."
    )


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class EmailState(TypedDict):
    customer_name: str
    customer_email: str
    subject: str
    message: str

    reasoning: str
    planning: str
    action: str
    observation: str
    final_response: str


# ============================================================
# 4. REASONING NODE
# ============================================================

def reasoning_node(state: EmailState):

    print("Running Reasoning Node...")

    prompt = f"""
You are an email support reasoning agent.

Analyze this customer email.

Customer:
{state['customer_name']}

Subject:
{state['subject']}

Message:
{state['message']}

Explain briefly:

1. What does the customer want?
2. Is an order ID present?
3. What action is required?

Do not invent information.
"""

    response = llm.invoke(prompt)

    return {
        "reasoning": response.content
    }


# ============================================================
# 5. PLANNING NODE
# ============================================================

def planning_node(state: EmailState):

    print("Running Planning Node...")

    prompt = f"""
You are an email support planning agent.

Customer email:
{state['message']}

Reasoning:
{state['reasoning']}

Create a short action plan.

If an order ID is present and the customer wants the
order status, the plan should be:

1. Retrieve the order status.
2. Observe the tool result.
3. Respond to the customer using the retrieved status.

Do not invent information.
"""

    response = llm.invoke(prompt)

    return {
        "planning": response.content
    }


# ============================================================
# 6. ACTION NODE
# ============================================================

def action_node(state: EmailState):

    print("Running Action Node...")

    message = state["message"]

    # Search for an order ID such as ORD123
    match = re.search(
        r"\bORD\d+\b",
        message.upper()
    )

    if match:

        order_id = match.group(0)

        print(f"Order ID detected: {order_id}")

        # Execute the LangChain tool
        result = get_order_status.invoke({
            "order_id": order_id
        })

        action_result = (
            f"Tool: get_order_status\n"
            f"Order ID: {order_id}\n"
            f"Tool Result: {result}"
        )

    else:

        action_result = (
            "No order ID was found in the email. "
            "The order status tool was not executed."
        )

    return {
        "action": action_result
    }


# ============================================================
# 7. OBSERVATION NODE
# ============================================================

def observation_node(state: EmailState):

    print("Running Observation Node...")

    observation = state["action"]

    return {
        "observation": observation
    }


# ============================================================
# 8. FINAL RESPONSE NODE
# ============================================================

def final_response_node(state: EmailState):

    print("Running Final Response Node...")

    # Keep this prompt small so llama3.2 responds quickly
    prompt = f"""
Write a short professional customer support email.

Customer name:
{state['customer_name']}

Customer's message:
{state['message']}

Tool result:
{state['observation']}

Rules:
- Use the actual tool result.
- Do not invent information.
- Do not invent dates.
- Do not invent tracking numbers.
- Do not invent delivery times.
- Do not include [Your Name].
- Keep the response under 80 words.
- End with:

Best regards,
Customer Support Team
"""

    response = llm.invoke(prompt)

    return {
        "final_response": response.content
    }


# ============================================================
# 9. CREATE LANGGRAPH
# ============================================================

graph = StateGraph(EmailState)


# Add nodes
graph.add_node(
    "reasoning",
    reasoning_node
)

graph.add_node(
    "planning",
    planning_node
)

graph.add_node(
    "action",
    action_node
)

graph.add_node(
    "observation",
    observation_node
)

graph.add_node(
    "final_response",
    final_response_node
)


# ============================================================
# 10. CONNECT NODES
# ============================================================

graph.add_edge(
    START,
    "reasoning"
)

graph.add_edge(
    "reasoning",
    "planning"
)

graph.add_edge(
    "planning",
    "action"
)

graph.add_edge(
    "action",
    "observation"
)

graph.add_edge(
    "observation",
    "final_response"
)

graph.add_edge(
    "final_response",
    END
)


# Compile the graph
app = graph.compile()


# ============================================================
# 11. TEST EMAIL
# ============================================================

initial_state = {

    "customer_name": "Ravi",

    "customer_email": "ravi@gmail.com",

    "subject": "Order Status",

    "message": (
        "Hello, I placed an order three days ago. "
        "My order number is ORD123. "
        "Please tell me the current status of my order."
    ),

    "reasoning": "",

    "planning": "",

    "action": "",

    "observation": "",

    "final_response": ""
}


# ============================================================
# 12. RUN LANGGRAPH WORKFLOW
# ============================================================

result = app.invoke(initial_state)


# ============================================================
# 13. DISPLAY RESULT
# ============================================================

print("\n")
print("=" * 60)
print("LANGGRAPH EMAIL WORKFLOW RESULT")
print("=" * 60)


print("\n🧠 REASONING")
print(result["reasoning"])


print("\n📋 PLANNING")
print(result["planning"])


print("\n⚡ ACTION")
print(result["action"])


print("\n👁️ OBSERVATION")
print(result["observation"])


print("\n📨 FINAL RESPONSE")
print(result["final_response"])


print("\n")
print("=" * 60)
print("LANGGRAPH WORKFLOW COMPLETED")
print("=" * 60)