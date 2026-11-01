import streamlit as st
import json

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# EXPERIMENT 3
# Email Classification and Structured Output using LangChain
# =========================================================


# ---------------------------------------------------------
# STEP 1: Define Structured Output Model
# ---------------------------------------------------------

class EmailClassification(BaseModel):

    intent: str = Field(
        description="The main intent of the email"
    )

    priority: str = Field(
        description="Priority of the email"
    )

    response_required: bool = Field(
        description="Whether a response is required"
    )

    action: str = Field(
        description="The action that should be taken"
    )


# ---------------------------------------------------------
# STEP 2: Create Ollama AI Model
# ---------------------------------------------------------

model = ChatOllama(
    model="llama3.2",
    temperature=0,
    format="json"
)


# ---------------------------------------------------------
# STEP 3: Create Email Classification Prompt
# ---------------------------------------------------------

classification_prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an Email Classification Agent.

Analyze the email provided by the user.

Determine the following:

1. The main intent of the email.
2. The priority of the email.
3. Whether a response is required.
4. The appropriate action that should be taken.

Priority must be exactly one of:

Low
Medium
High
Urgent

The response_required field must be either true or false.

Return ONLY valid JSON.

Use exactly this JSON structure:

{{
    "intent": "main intent of the email",
    "priority": "Low/Medium/High/Urgent",
    "response_required": true,
    "action": "action that should be taken"
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


# ---------------------------------------------------------
# STEP 4: Create LangChain Classification Chain
# ---------------------------------------------------------

classifier = classification_prompt | model


# ---------------------------------------------------------
# STEP 5: Configure Streamlit Page
# ---------------------------------------------------------

st.set_page_config(
    page_title="Email Classifier - Experiment 3",
    page_icon="📧",
    layout="centered"
)


# ---------------------------------------------------------
# STEP 6: Application Title
# ---------------------------------------------------------

st.title("📧 Email Classifier")

st.header("Experiment 3")

st.write(
    "Email Classification and Structured Output using "
    "LangChain and Ollama"
)


# ---------------------------------------------------------
# STEP 7: About Experiment
# ---------------------------------------------------------

with st.expander("📚 About Experiment 3"):

    st.write(
        "This experiment uses LangChain and an Ollama "
        "LLM to classify incoming emails."
    )

    st.write(
        "The Email Classification Agent identifies:"
    )

    st.write("• Email Intent")
    st.write("• Email Priority")
    st.write("• Whether a Response is Required")
    st.write("• Required Action")


# ---------------------------------------------------------
# STEP 8: Email Details
# ---------------------------------------------------------

st.subheader("📨 Email Details")


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
    placeholder="Example: Order Status"
)


message = st.text_area(
    "Email Message",
    height=180,
    placeholder=(
        "Example: I placed an order three days ago. "
        "Please tell me the current status of my order."
    )
)


# ---------------------------------------------------------
# STEP 9: Classify Email Button
# ---------------------------------------------------------

if st.button("🤖 Classify Email"):

    # -----------------------------------------------------
    # Check whether email message is entered
    # -----------------------------------------------------

    if not message.strip():

        st.warning(
            "Please enter an email message."
        )

    else:

        # -------------------------------------------------
        # Display Agent Workflow
        # -------------------------------------------------

        st.subheader("🔄 Agent Workflow")

        st.write(
            "Step 1 : Email Received"
        )

        st.write(
            "Step 2 : Email Sent to Classification Agent"
        )

        st.write(
            "Step 3 : Agent Understanding Email"
        )

        st.write(
            "Step 4 : Identifying Email Intent"
        )

        st.write(
            "Step 5 : Determining Email Priority"
        )

        st.write(
            "Step 6 : Determining Response Requirement"
        )

        st.write(
            "Step 7 : Determining Required Action"
        )


        # -------------------------------------------------
        # STEP 10: Run AI Agent
        # -------------------------------------------------

        try:

            with st.spinner(
                "🤖 Llama 3.2 is analyzing the email..."
            ):

                response = classifier.invoke({

                    "name": name,

                    "email": email,

                    "subject": subject,

                    "message": message

                })


            # -------------------------------------------------
            # STEP 11: Get AI Response
            # -------------------------------------------------

            raw_result = response.content


            # -------------------------------------------------
            # Remove Markdown Code Fences if AI adds them
            # -------------------------------------------------

            raw_result = raw_result.strip()

            if raw_result.startswith("```json"):

                raw_result = raw_result[
                    len("```json"):
                ]

                raw_result = raw_result.replace(
                    "```",
                    ""
                ).strip()

            elif raw_result.startswith("```"):

                raw_result = raw_result[
                    len("```"):
                ]

                raw_result = raw_result.replace(
                    "```",
                    ""
                ).strip()


            # -------------------------------------------------
            # STEP 12: Convert JSON to Python Dictionary
            # -------------------------------------------------

            result_dict = json.loads(
                raw_result
            )


            # -------------------------------------------------
            # STEP 13: Validate with Pydantic
            # -------------------------------------------------

            result = EmailClassification.model_validate(
                result_dict
            )


            # -------------------------------------------------
            # STEP 14: Display Classification Result
            # -------------------------------------------------

            st.subheader(
                "📊 Email Classification Result"
            )

            st.success(
                "Email successfully classified!"
            )


            # -------------------------------------------------
            # Display Intent
            # -------------------------------------------------

            st.write(
                "**Intent:**",
                result.intent
            )


            # -------------------------------------------------
            # Display Priority
            # -------------------------------------------------

            st.write(
                "**Priority:**",
                result.priority
            )


            # -------------------------------------------------
            # Display Response Requirement
            # -------------------------------------------------

            if result.response_required:

                st.write(
                    "**Response Required:** Yes"
                )

            else:

                st.write(
                    "**Response Required:** No"
                )


            # -------------------------------------------------
            # Display Action
            # -------------------------------------------------

            st.write(
                "**Action:**",
                result.action
            )


            # -------------------------------------------------
            # STEP 15: Display Structured JSON
            # -------------------------------------------------

            st.subheader(
                "🧾 Structured JSON Output"
            )

            st.json(
                result.model_dump()
            )


            # -------------------------------------------------
            # STEP 16: Display Raw JSON
            # -------------------------------------------------

            with st.expander("🔍 View Raw AI Response"):

                st.code(
                    raw_result,
                    language="json"
                )


            # -------------------------------------------------
            # STEP 17: Experiment Completion
            # -------------------------------------------------

            st.success(
                "✅ Experiment 3 Completed Successfully!"
            )


        # -----------------------------------------------------
        # JSON Error
        # -----------------------------------------------------

        except json.JSONDecodeError:

            st.error(
                "The AI returned an invalid JSON response."
            )

            st.write(
                "Raw AI Response:"
            )

            st.code(
                raw_result
            )


        # -----------------------------------------------------
        # General Error
        # -----------------------------------------------------

        except Exception as e:

            st.error(
                "An error occurred while processing the email."
            )

            st.exception(e)