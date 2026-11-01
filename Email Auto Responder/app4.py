import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Ollama model
model = ChatOllama(
    model="llama3.2",
    temperature=0,
    format="json"
)

# Email Filter Agent prompt
filter_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Email Filter Agent.

Your job is to examine an incoming email and decide
whether the email is important and needs attention.

Important emails include:
- Customer complaints
- Order problems
- Payment problems
- Urgent requests
- Account problems
- Requests that require a response

Low-priority emails include:
- Advertisements
- Newsletters
- Promotional emails
- Spam-like messages
- Emails that do not require action

Return ONLY valid JSON.

Use exactly this structure:

{{
    "important": true,
    "reason": "Reason for the decision"
}}
"""
    ),
    (
        "human",
        """
Sender Name: {name}
Sender Email: {email}
Subject: {subject}

Email Message:
{message}
"""
    )
])

# Create Agent 1
filter_agent = filter_prompt | model
# Test Agent 1
if __name__ == "__main__":
    response = filter_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order."
    })

    print("\n===== EMAIL FILTER AGENT =====")
    print(response.content)
    # Email Action Agent prompt
action_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Email Action Agent.

You receive an important email that has already been
identified by the Email Filter Agent.

Your job is to determine:

1. Whether an action is required.
2. What action should be taken.
3. The priority of the email.

Priority must be exactly one of:

Low
Medium
High
Urgent

Return ONLY valid JSON.

Use exactly this structure:

{{
    "action_required": true,
    "action": "Action that should be taken",
    "priority": "Medium"
}}
"""
    ),
    (
        "human",
        """
Sender Name: {name}
Sender Email: {email}
Subject: {subject}

Email Message:
{message}

Email Filter Agent Decision:
{filter_result}
"""
    )
])

# Create Agent 2
action_agent = action_prompt | model
# Test Agent 2
if __name__ == "__main__":
    # Step 1: Agent 1 filters the email
    filter_response = filter_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order."
    })

    print("\n===== AGENT 1: EMAIL FILTER =====")
    print(filter_response.content)

    # Step 2: Agent 2 decides the required action
    action_response = action_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order.",
        "filter_result": filter_response.content
    })

    print("\n===== AGENT 2: EMAIL ACTION =====")
    print(action_response.content)
    # Email Response Writer Agent prompt
response_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an Email Response Writer Agent.

Your job is to write a professional and polite email
response based on the original email and the action
identified by the Email Action Agent.

The response should:
- Be professional
- Be polite
- Directly address the customer's request
- Be concise
- Not invent information
- Not claim that an action was completed if it was
  not actually completed

Return ONLY valid JSON.

Use exactly this structure:

{{
    "subject": "Reply subject",
    "message": "Professional email response"
}}
"""
    ),
    (
        "human",
        """
Original Email:

Sender Name: {name}
Sender Email: {email}
Subject: {subject}

Message:
{message}

Email Action Agent Decision:
{action_result}
"""
    )
])

# Create Agent 3
response_agent = response_prompt | model
if __name__ == "__main__":

    # Step 1: Agent 1 - Filter the email
    filter_response = filter_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order."
    })

    print("\n===== AGENT 1: EMAIL FILTER =====")
    print(filter_response.content)

    # Step 2: Agent 2 - Decide the action
    action_response = action_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order.",
        "filter_result": filter_response.content
    })

    print("\n===== AGENT 2: EMAIL ACTION =====")
    print(action_response.content)

    # Step 3: Agent 3 - Write the response
    response_response = response_agent.invoke({
        "name": "Ravi",
        "email": "ravi@gmail.com",
        "subject": "Order Problem",
        "message": "Hello, my order has not arrived yet. I placed the order five days ago and the delivery is already delayed. Please help me check the status of my order.",
        "action_result": action_response.content
    })

    print("\n===== AGENT 3: RESPONSE WRITER =====")
    print(response_response.content)
    # ==============================
# STREAMLIT INTERFACE
# ==============================

st.set_page_config(
    page_title="Multi-Agent Email System",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Multi-Agent Email System")

st.subheader("Experiment 4")
st.write("Building a Multi-Agent Email System using LangChain and Ollama")

st.divider()

st.subheader("📨 Enter Email Details")

name = st.text_input(
    "Sender Name",
    placeholder="Example: Ravi"
)

email = st.text_input(
    "Sender Email",
    placeholder="Example: ravi@gmail.com"
)

subject = st.text_input(
    "Email Subject",
    placeholder="Example: Order Problem"
)

message = st.text_area(
    "Email Message",
    height=180,
    placeholder="Enter the email message here..."
)

if st.button("🤖 Process Email"):

    if not message.strip():
        st.warning("Please enter an email message.")

    else:

        st.subheader("🔄 Multi-Agent Workflow")

        with st.spinner("🤖 Agents are processing the email..."):

            # Agent 1
            filter_response = filter_agent.invoke({
                "name": name,
                "email": email,
                "subject": subject,
                "message": message
            })

            # Agent 2
            action_response = action_agent.invoke({
                "name": name,
                "email": email,
                "subject": subject,
                "message": message,
                "filter_result": filter_response.content
            })

            # Agent 3
            response_response = response_agent.invoke({
                "name": name,
                "email": email,
                "subject": subject,
                "message": message,
                "action_result": action_response.content
            })

        st.success("✅ Email processing completed!")

        st.subheader("🤖 Agent 1 — Email Filter")

        st.code(filter_response.content, language="json")

        st.subheader("⚙️ Agent 2 — Email Action")

        st.code(action_response.content, language="json")

        st.subheader("✍️ Agent 3 — Response Writer")

        st.code(response_response.content, language="json")

        st.subheader("📨 Final Email Draft")

        st.write(response_response.content)