import streamlit as st
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

    prompt = f"""
You are an email support planning agent.

Customer email:
{state['message']}

Reasoning:
{state['reasoning']}

Create a short action plan.

If an order ID is present and the customer wants
the order status:

1. Retrieve the order status.
2. Observe the tool result.
3. Respond using the retrieved status.

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

    message = state["message"]

    # Find order ID such as ORD123, ORD456, etc.
    match = re.search(
        r"\bORD\d+\b",
        message.upper()
    )

    if match:

        order_id = match.group(0)

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

    return {
        "observation": state["action"]
    }


# ============================================================
# 8. FINAL RESPONSE NODE
# ============================================================

def final_response_node(state: EmailState):

    prompt = f"""
Write a short professional customer support email.

Customer name:
{state['customer_name']}

Customer's message:
{state['message']}

Tool result:
{state['observation']}

Rules:

- Use ONLY information from the customer message
  and tool result.
- Do not invent information.
- Do not invent dates.
- Do not invent tracking numbers.
- Do not invent delivery times.
- Do not claim information that is not present.
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
# 9. BUILD LANGGRAPH
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
# 10. CONNECT LANGGRAPH NODES
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


# Compile graph
app = graph.compile()


# ============================================================
# 11. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LangGraph Email Auto Responder",
    page_icon="📧",
    layout="wide"
)


# ============================================================
# 12. STREAMLIT HEADER
# ============================================================

st.title("📧 LangGraph Email Auto Responder")

st.write(
    "Experiment 6 — Building an Email Workflow using LangGraph"
)

st.divider()


# ============================================================
# 13. CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Information")

customer_name = st.text_input(
    "Customer Name",
    value="Ravi"
)

customer_email = st.text_input(
    "Customer Email",
    value="ravi@gmail.com"
)

subject = st.text_input(
    "Subject",
    value="Order Status"
)

message = st.text_area(
    "Email Message",
    value=(
        "Hello, I placed an order three days ago. "
        "My order number is ORD123. "
        "Please tell me the current status of my order."
    ),
    height=160
)


# ============================================================
# 14. WORKFLOW INFORMATION
# ============================================================

st.subheader("🔄 LangGraph Workflow")

st.info(
    "START → Reasoning → Planning → Action → "
    "Observation → Final Response → END"
)


# ============================================================
# 15. RUN BUTTON
# ============================================================

if st.button(
    "🚀 Run Email Agent",
    use_container_width=True
):

    # Validate input
    if not customer_name.strip():
        st.error("Please enter the customer name.")

    elif not customer_email.strip():
        st.error("Please enter the customer email.")

    elif not subject.strip():
        st.error("Please enter the email subject.")

    elif not message.strip():
        st.error("Please enter the email message.")

    else:

        # Initial LangGraph state
        initial_state = {

            "customer_name": customer_name,

            "customer_email": customer_email,

            "subject": subject,

            "message": message,

            "reasoning": "",

            "planning": "",

            "action": "",

            "observation": "",

            "final_response": ""
        }


        # Run workflow
        with st.spinner(
            "Running LangGraph email workflow..."
        ):

            result = app.invoke(
                initial_state
            )


        st.success(
            "Email workflow completed successfully!"
        )


        # ====================================================
        # REASONING
        # ====================================================

        st.subheader("🧠 Reasoning")

        st.write(
            result["reasoning"]
        )


        # ====================================================
        # PLANNING
        # ====================================================

        st.subheader("📋 Planning")

        st.write(
            result["planning"]
        )


        # ====================================================
        # ACTION
        # ====================================================

        st.subheader("⚡ Action")

        st.code(
            result["action"]
        )


        # ====================================================
        # OBSERVATION
        # ====================================================

        st.subheader("👁️ Observation")

        st.code(
            result["observation"]
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        st.subheader("📨 Final Response")

        st.text_area(
            "Generated Customer Email",
            value=result["final_response"],
            height=180
        )