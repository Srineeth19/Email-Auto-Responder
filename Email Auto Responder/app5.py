import streamlit as st
import json
import re

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# ============================================================
# EXPERIMENT 5
# Intelligent Email Agent with Custom LangChain Tools
#
# Reasoning → Planning → Action → Observation → Final Response
# ============================================================


# ============================================================
# STEP 1 — INITIALIZE LOCAL LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# STEP 2 — CREATE CUSTOM LANGCHAIN TOOLS
# ============================================================

@tool
def get_order_status(order_id: str) -> str:
    """
    Check the current status of a customer's order
    using the order ID.
    """

    order_database = {
        "ORD123": "Shipped",
        "ORD456": "Processing",
        "ORD789": "Delivered",
        "ORD101": "Cancelled"
    }

    order_id = order_id.strip().upper()

    return order_database.get(
        order_id,
        "Order ID not found."
    )


@tool
def get_customer_information(email: str) -> dict:
    """
    Retrieve customer information using the customer's
    email address.
    """

    customer_database = {
        "ravi@gmail.com": {
            "name": "Ravi",
            "customer_id": "CUS001",
            "membership": "Premium"
        },
        "arun@gmail.com": {
            "name": "Arun",
            "customer_id": "CUS002",
            "membership": "Regular"
        }
    }

    email = email.strip().lower()

    customer = customer_database.get(email)

    if customer:
        return customer

    return {
        "error": "Customer not found."
    }


@tool
def get_order_support_information(topic: str) -> str:
    """
    Retrieve general customer support information.
    """

    support_information = {
        "shipping": "Orders marked as Shipped have been dispatched.",
        "processing": "Orders marked as Processing are being prepared.",
        "delivered": "Orders marked as Delivered have reached the customer.",
        "cancelled": "Orders marked as Cancelled are no longer active."
    }

    topic = topic.strip().lower()

    return support_information.get(
        topic,
        "No support information available."
    )


# ============================================================
# STEP 3 — REGISTER TOOLS
# ============================================================

tools = [
    get_order_status,
    get_customer_information,
    get_order_support_information
]

tools_by_name = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# STEP 4 — EXTRACT ORDER ID
# ============================================================

def extract_order_id(message: str):

    pattern = r"\bORD\d+\b"

    match = re.search(
        pattern,
        message.upper()
    )

    if match:
        return match.group(0)

    return None


# ============================================================
# STEP 5 — PROCESS EMAIL
# ============================================================

def process_email(
    name,
    email,
    subject,
    message
):

    # ========================================================
    # 5.1 REASONING
    # ========================================================

    reasoning_prompt = f"""
You are an intelligent customer support email agent.

Analyze the customer email.

Customer Name:
{name}

Customer Email:
{email}

Subject:
{subject}

Message:
{message}

Explain:

1. What the customer wants.
2. What information is required.
3. Whether an order number is present.

IMPORTANT:
- Use only information present in the email.
- Do not invent an order number.
- Do not assume an order status.
"""

    reasoning_response = llm.invoke(
        reasoning_prompt
    )

    reasoning = reasoning_response.content


    # ========================================================
    # 5.2 DETECT ORDER ID
    # ========================================================

    order_id = extract_order_id(message)


    # ========================================================
    # 5.3 PLANNING
    # ========================================================

    if order_id:

        planning_prompt = f"""
You are an AI customer support agent.

Customer Name:
{name}

Customer Email:
{email}

Subject:
{subject}

Message:
{message}

Detected Order ID:
{order_id}

Create a short action plan.

The customer is asking about an order status.

MANDATORY ACTION:

Use:

get_order_status

with:

order_id = {order_id}

The order status must be obtained from the order status tool.

Do not invent information.
Do not create additional order information.
"""

    else:

        planning_prompt = f"""
You are an AI customer support agent.

Customer Name:
{name}

Customer Email:
{email}

Subject:
{subject}

Message:
{message}

No order ID was detected.

Create a short action plan.

If the customer is asking about order status,
do not invent an order ID.

The final response should ask the customer
to provide their order number.

Do not invent information.
"""


    planning_response = llm.invoke(
        planning_prompt
    )

    planning = planning_response.content


    # ========================================================
    # 5.4 ACTION
    # ========================================================

    tool_calls_display = []
    tool_results = []


    if order_id:

        # ----------------------------------------------------
        # FORCE THE CORRECT TOOL
        # ----------------------------------------------------
        # If an order ID exists, the order-status tool
        # must be executed.
        # ----------------------------------------------------

        tool_call = {
            "name": "get_order_status",
            "args": {
                "order_id": order_id
            }
        }


        tool_calls_display.append({
            "name": tool_call["name"],
            "args": tool_call["args"]
        })


        # Get the LangChain tool
        selected_tool = tools_by_name.get(
            tool_call["name"]
        )


        if selected_tool is None:

            raise ValueError(
                "get_order_status tool was not found."
            )


        # ----------------------------------------------------
        # IMPORTANT:
        # Pass ONLY the arguments to invoke().
        #
        # Correct:
        # {"order_id": "ORD123"}
        #
        # NOT the entire tool_call object.
        # ----------------------------------------------------

        tool_result = selected_tool.invoke(
            tool_call["args"]
        )


        # ----------------------------------------------------
        # Convert result to JSON-safe value
        # ----------------------------------------------------

        if hasattr(tool_result, "content"):

            result_content = tool_result.content

        else:

            result_content = tool_result


        # ----------------------------------------------------
        # Store observation
        # ----------------------------------------------------

        tool_results.append({
            "tool": tool_call["name"],
            "arguments": tool_call["args"],
            "result": result_content
        })


    else:

        tool_calls_display = []
        tool_results = []


    # ========================================================
    # 5.5 OBSERVATION
    # ========================================================

    # The tool result has been stored in tool_results.
    #
    # Example:
    #
    # Tool:
    # get_order_status
    #
    # Arguments:
    # order_id = ORD123
    #
    # Result:
    # Shipped


    # ========================================================
    # 5.6 FINAL RESPONSE
    # ========================================================

    if order_id:

        # ----------------------------------------------------
        # Get the VERIFIED status directly from the tool
        # ----------------------------------------------------

        verified_status = tool_results[0]["result"]


        final_prompt = f"""
You are a customer support email formatter.

Write a short and professional email.

Customer Name:
{name}

Order Number:
{order_id}

VERIFIED ORDER STATUS:
{verified_status}


STRICT RULES
============

1. The order status MUST be taken directly from:

{verified_status}

2. Do not change the order status.

3. Do not invent any information.

4. Do not assume information that is not provided.

5. Do NOT invent:

- shipping dates
- delivery dates
- delivery timeframes
- tracking numbers
- estimated delivery dates
- prices
- addresses
- phone numbers
- refund amounts
- company policies
- system update times

6. Do NOT use placeholders.

Never write:

[Your Name]
[Date]
[Timeframe]
[Tracking Number]

7. Do NOT say the information is "up-to-date".

8. Do NOT claim that the order was "located" unless this
is explicitly provided.

9. Do NOT claim that the customer was "verified".

10. Do NOT mention internal tools.

11. Do NOT mention the AI agent.

12. Do NOT mention reasoning or planning.

13. Address the customer as:

{name}

14. Include the order number:

{order_id}

15. State the verified order status:

{verified_status}

16. Keep the email short and professional.

17. Return ONLY the customer-facing email.

Use this structure:

Subject: Re: Order Status

Dear {name},

The current status of your order {order_id} is
{verified_status}.

Best regards,
Customer Support Team
"""


        final_response = llm.invoke(
            final_prompt
        )

        final_email = final_response.content


    else:

        # ----------------------------------------------------
        # NO ORDER ID
        # ----------------------------------------------------

        final_prompt = f"""
You are a professional customer support email agent.

Customer Name:
{name}

Customer Email:
{email}

Subject:
{subject}

Message:
{message}

The customer is asking about an order, but no order ID
was provided.

Write a short professional email asking the customer
to provide their order number.

STRICT RULES:

- Do not invent an order number.
- Do not invent an order status.
- Do not claim that the order was checked.
- Do not provide a delivery estimate.
- Do not use placeholders.
- Do not mention internal tools.
- Do not mention reasoning.
- Return ONLY the customer-facing email.
"""


        final_response = llm.invoke(
            final_prompt
        )

        final_email = final_response.content


    # ========================================================
    # COMPLETE AGENT OUTPUT
    # ========================================================

    complete_output = {

        "reasoning": reasoning,

        "planning": planning,

        "tool_calls": tool_calls_display,

        "tool_results": tool_results,

        "final_response": final_email

    }


    return complete_output


# ============================================================
# STEP 6 — STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Email Agent",
    page_icon="📧",
    layout="wide"
)


# ============================================================
# STEP 7 — APPLICATION HEADER
# ============================================================

st.title("📧 Intelligent Email Agent")

st.subheader("Experiment 5")

st.write(
    "Reasoning → Planning → Action → Observation → Final Response"
)


# ============================================================
# STEP 8 — CUSTOMER EMAIL
# ============================================================

st.markdown("## 📨 Customer Email")


customer_name = st.text_input(
    "Customer Name",
    value="Ravi"
)


customer_email = st.text_input(
    "Customer Email",
    value="ravi@gmail.com"
)


email_subject = st.text_input(
    "Email Subject",
    value="Order Status"
)


email_message = st.text_area(
    "Email Message",
    value=(
        "Hello, I placed an order three days ago. "
        "The order number is ORD123. "
        "Please tell me the current status of my order."
    ),
    height=150
)


# ============================================================
# STEP 9 — PROCESS EMAIL
# ============================================================

if st.button(
    "🚀 Process Email",
    type="primary"
):

    with st.spinner(
        "Intelligent Agent is processing the email..."
    ):

        try:

            result = process_email(
                customer_name,
                customer_email,
                email_subject,
                email_message
            )


            # =================================================
            # 1. REASONING
            # =================================================

            st.markdown(
                "## 🧠 1. Reasoning"
            )

            st.write(
                result["reasoning"]
            )


            # =================================================
            # 2. PLANNING
            # =================================================

            st.markdown(
                "## 📋 2. Planning"
            )

            st.write(
                result["planning"]
            )


            # =================================================
            # 3. ACTION
            # =================================================

            st.markdown(
                "## ⚡ 3. Action"
            )


            if result["tool_calls"]:

                for call in result["tool_calls"]:

                    st.write(
                        f"**Selected Tool:** "
                        f"{call['name']}"
                    )

                    st.write(
                        "**Arguments:**"
                    )

                    st.json(
                        call["args"]
                    )

            else:

                st.info(
                    "No tool was executed because "
                    "no order ID was provided."
                )


            # =================================================
            # 4. OBSERVATION
            # =================================================

            st.markdown(
                "## 👁️ 4. Observation"
            )


            if result["tool_results"]:

                for observation in result["tool_results"]:

                    st.write(
                        f"**Tool:** "
                        f"{observation['tool']}"
                    )

                    st.write(
                        "**Arguments:**"
                    )

                    st.json(
                        observation["arguments"]
                    )

                    st.write(
                        f"**Tool Result:** "
                        f"{observation['result']}"
                    )

            else:

                st.info(
                    "No tool result available."
                )


            # =================================================
            # 5. FINAL RESPONSE
            # =================================================

            st.markdown(
                "## 📨 5. Final Email Response"
            )

            st.text(
                result["final_response"]
            )


            # =================================================
            # COMPLETE OUTPUT
            # =================================================

            with st.expander(
                "🔎 View Complete Agent Output"
            ):

                st.json(
                    result
                )


        except Exception as e:

            st.error(
                f"An error occurred: {str(e)}"
            )