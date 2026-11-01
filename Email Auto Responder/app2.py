import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# EXPERIMENT 2
# Single Email Assistant using LangChain
# =========================================================


# ---------------------------------------------------------
# STEP 1: Create AI Model
# ---------------------------------------------------------

model = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ---------------------------------------------------------
# STEP 2: Create Email Agent Prompt
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an intelligent Email Assistant.

Your job is to analyze the email provided by the user.

Perform the following tasks:

1. Identify the email intent.
2. Determine the priority.
3. Decide whether a response is required.
4. Decide what action should be taken.
5. Generate a professional email response.

Possible priorities:
- Low
- Medium
- High
- Urgent

Keep your answer professional, clear and concise.

Use this format:

Intent:
Priority:
Response Required:
Action:
Response:
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


# ---------------------------------------------------------
# STEP 3: Streamlit Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="Email Assistant - Experiment 2",
    page_icon="📧"
)

st.title("📧 Email Assistant - Experiment 2")

st.write(
    "Single Agent Email Assistant using LangChain and Ollama"
)


# ---------------------------------------------------------
# STEP 4: Email Details
# ---------------------------------------------------------

st.subheader("📨 Email Details")

name = st.text_input(
    "Enter Your Name"
)

email = st.text_input(
    "Enter Your Email Address"
)

subject = st.text_input(
    "Enter Email Subject"
)

message = st.text_area(
    "Enter Your Email Message",
    height=180
)


# ---------------------------------------------------------
# STEP 5: Process Email
# ---------------------------------------------------------

if st.button("Process Email"):

    if not message.strip():

        st.warning(
            "Please enter an email message."
        )

    else:

        st.subheader("🤖 Agent Workflow")

        # -------------------------------------------------
        # STEP 6: Show Workflow
        # -------------------------------------------------

        st.write(
            "Step 1 : Email Received"
        )

        st.write(
            "Step 2 : Email Sent to AI Agent"
        )

        st.write(
            "Step 3 : Agent Understanding Email"
        )

        st.write(
            "Step 4 : Identifying Email Intent"
        )

        st.write(
            "Step 5 : Determining Priority"
        )

        st.write(
            "Step 6 : Deciding Required Action"
        )


        # -------------------------------------------------
        # STEP 7: Create Prompt
        # -------------------------------------------------

        formatted_prompt = prompt.invoke({
            "name": name,
            "email": email,
            "subject": subject,
            "message": message
        })


        # -------------------------------------------------
        # STEP 8: Send Email to AI Model
        # -------------------------------------------------

        with st.spinner("🤖 AI Agent is analyzing the email..."):

            response = model.invoke(
                formatted_prompt
            )


        # -------------------------------------------------
        # STEP 9: Display Result
        # -------------------------------------------------

        st.write(
            "Step 7 : Generating Email Response"
        )

        st.subheader(
            "📨 Agent Result"
        )

        st.write(
            response.content
        )


        # -------------------------------------------------
        # STEP 10: Completion
        # -------------------------------------------------

        st.success(
            "Email Processing Completed Successfully!"
        )