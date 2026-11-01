import re
from pathlib import Path
from typing import TypedDict

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG-Based Email Auto-Responder | Experiment 7",
    page_icon="📧",
    layout="wide",
)

st.title("📧 RAG-Based Email Auto-Responder")
st.caption(
    "Experiment 7 — Retrieval-Augmented Generation using "
    "LangChain, FAISS, Ollama, and LangGraph"
)

st.markdown(
    """
    **Workflow**

    START → RAG Retrieval → Reasoning → Planning → Action
    → Observation → Final Response → END
    """
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = (
    BASE_DIR / "knowledge_base" / "email_support_knowledge.txt"
)

MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ORDER_DATABASE = {
    "ORD123": "Shipped",
    "ORD456": "Processing",
    "ORD789": "Delivered",
    "ORD101": "Cancelled",
}


# ============================================================
# LOAD EMBEDDINGS AND BUILD FAISS INDEX
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model and knowledge base...")
def load_knowledge_base():
    """Load support policies and create a searchable FAISS index."""

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge-base file was not found: {KNOWLEDGE_FILE}\n"
            "Create the knowledge_base folder and add "
            "email_support_knowledge.txt."
        )

    loader = TextLoader(
        str(KNOWLEDGE_FILE),
        encoding="utf-8",
    )
    documents = loader.load()

    if not documents:
        raise ValueError("The knowledge-base file is empty.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the knowledge base.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


# ============================================================
# LOAD OLLAMA LANGUAGE MODEL
# ============================================================

@st.cache_resource
def load_llm():
    """Create the local Ollama language model."""

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )


# ============================================================
# ORDER STATUS TOOL
# ============================================================

def get_order_status(order_id: str) -> str:
    """
    Mock order-status tool.

    In a real application, replace this dictionary lookup with
    an approved order-management system or API.
    """

    normalized_id = order_id.strip().upper()

    if normalized_id in ORDER_DATABASE:
        return ORDER_DATABASE[normalized_id]

    return "Order not found in the mock order database."


# ============================================================
# LANGGRAPH STATE
# ============================================================

class EmailState(TypedDict, total=False):
    customer_name: str
    customer_email: str
    subject: str
    email_body: str

    order_id: str
    intent: str

    retrieved_documents: list[str]
    reasoning: str
    planning: str

    action: str
    tool_result: str
    observation: str

    final_response: str
    error: str


# ============================================================
# NODE 1: RAG RETRIEVAL
# ============================================================

def rag_retrieval(state: EmailState) -> dict:
    """Retrieve support knowledge relevant to the customer's email."""

    query = (
        f"Subject: {state['subject']}\n"
        f"Customer email: {state['email_body']}"
    )

    try:
        vector_store = load_knowledge_base()
        documents = vector_store.similarity_search(query, k=3)

        retrieved_texts = [
            document.page_content for document in documents
        ]

        return {
            "retrieved_documents": retrieved_texts,
            "error": "",
        }

    except Exception as exc:
        return {
            "retrieved_documents": [],
            "error": f"Knowledge retrieval failed: {exc}",
        }


# ============================================================
# NODE 2: REASONING
# ============================================================

def reasoning_node(state: EmailState) -> dict:
    """Identify intent and determine what information is required."""

    email_text = (
        f"{state['subject']}\n{state['email_body']}"
    ).lower()

    order_match = re.search(
        r"\bORD\d+\b",
        email_text,
        flags=re.IGNORECASE,
    )

    order_id = (
        order_match.group(0).upper()
        if order_match
        else ""
    )

    if any(
        word in email_text
        for word in ["order status", "track my order",
                     "where is my order", "delivery status",
                     "order update", "shipped"]
    ):
        intent = "Order status enquiry"

    elif any(
        word in email_text
        for word in ["cancel", "cancellation"]
    ):
        intent = "Order cancellation enquiry"

    elif any(
        word in email_text
        for word in ["refund", "money back", "reimbursement"]
    ):
        intent = "Refund enquiry"

    elif any(
        word in email_text
        for word in ["return", "exchange", "replace"]
    ):
        intent = "Return or exchange enquiry"

    elif any(
        word in email_text
        for word in ["password", "account", "login", "sign in"]
    ):
        intent = "Account support enquiry"

    else:
        intent = "General customer support enquiry"

    if intent == "Order status enquiry":
        if order_id:
            action_needed = (
                f"Verify order {order_id} using the order-status tool."
            )
        else:
            action_needed = (
                "Ask the customer to provide their order ID."
            )

    elif intent == "Order cancellation enquiry":
        action_needed = (
            "Review the cancellation policy. Do not promise "
            "cancellation without verification."
        )

    elif intent == "Refund enquiry":
        action_needed = (
            "Review the approved refund policy. Do not claim "
            "that a refund was issued without verification."
        )

    elif intent == "Return or exchange enquiry":
        action_needed = (
            "Review the available return policy and recommend "
            "support-team verification if details are missing."
        )

    elif intent == "Account support enquiry":
        action_needed = (
            "Review the available account-support guidance. "
            "Do not invent customer account information."
        )

    else:
        action_needed = (
            "Use the retrieved support knowledge to prepare "
            "a helpful response."
        )

    reasoning = (
        f"Customer intent: {intent}\n"
        f"Order ID present: {'Yes (' + order_id + ')' if order_id else 'No'}\n"
        f"Action needed: {action_needed}"
    )

    return {
        "intent": intent,
        "order_id": order_id,
        "reasoning": reasoning,
    }


# ============================================================
# NODE 3: PLANNING
# ============================================================

def planning_node(state: EmailState) -> dict:
    """Plan the next step based on intent and available information."""

    intent = state.get("intent", "")
    order_id = state.get("order_id", "")

    if intent == "Order status enquiry":
        if order_id:
            planning = (
                f"Look up order {order_id} using get_order_status. "
                "Include only the verified status in the email."
            )
        else:
            planning = (
                "The order ID is missing. Ask the customer to "
                "provide it before checking the order status."
            )

    elif intent == "Order cancellation enquiry":
        planning = (
            "Review the retrieved cancellation policy and "
            "recommend verification if eligibility is unclear."
        )

    elif intent == "Refund enquiry":
        planning = (
            "Review the retrieved refund policy. Do not invent "
            "refund timelines or claim a refund has been issued."
        )

    elif intent == "Return or exchange enquiry":
        planning = (
            "Use the retrieved return policy and explain that "
            "the support team must verify missing details."
        )

    elif intent == "Account support enquiry":
        planning = (
            "Use the retrieved account-support information. "
            "Do not invent customer-specific information."
        )

    else:
        planning = (
            "Answer using the retrieved knowledge and recommend "
            "support-team verification for unanswered questions."
        )

    return {"planning": planning}


# ============================================================
# NODE 4: ACTION
# ============================================================

def action_node(state: EmailState) -> dict:
    """Execute the appropriate tool action when required."""

    intent = state.get("intent", "")
    order_id = state.get("order_id", "")

    if intent == "Order status enquiry" and order_id:
        tool_result = get_order_status(order_id)

        action = (
            "Action: Order Status Lookup\n"
            "Tool: get_order_status\n"
            f"Order ID: {order_id}\n"
            f"Tool Result: {tool_result}"
        )

    elif intent == "Order status enquiry" and not order_id:
        tool_result = "Not executed: order ID is missing."

        action = (
            "Action: Order Status Lookup\n"
            "Tool: get_order_status\n"
            "Tool Result: Not executed because the order ID is missing."
        )

    else:
        tool_result = (
            "No external tool was required for this request."
        )

        action = (
            "Action: Knowledge-based response\n"
            "Tool: Not required\n"
            f"Details: {state.get('planning', '')}"
        )

    return {
        "action": action,
        "tool_result": tool_result,
    }


# ============================================================
# NODE 5: OBSERVATION
# ============================================================

def observation_node(state: EmailState) -> dict:
    """Record the real tool result and retrieved policy information."""

    order_id = state.get("order_id", "")
    tool_result = state.get("tool_result", "")
    documents = state.get("retrieved_documents", [])

    if order_id and state.get("intent") == "Order status enquiry":
        observed_tool = (
            f"Order ID: {order_id}\n"
            f"Verified tool result: {tool_result}"
        )
    else:
        observed_tool = (
            f"Tool result: {tool_result or 'No tool result available.'}"
        )

    if documents:
        knowledge_summary = "\n\n".join(
            f"Knowledge {index + 1}: {document}"
            for index, document in enumerate(documents)
        )
    else:
        knowledge_summary = (
            "No knowledge documents were retrieved. "
            "The response must not invent policy details."
        )

    observation = (
        f"{observed_tool}\n\n"
        f"Retrieved support knowledge:\n{knowledge_summary}"
    )

    return {"observation": observation}


# ============================================================
# NODE 6: FINAL RESPONSE
# ============================================================

def final_response_node(state: EmailState) -> dict:
    """Generate a professional customer email using verified evidence."""

    name = state.get("customer_name", "").strip() or "Customer"
    subject = state.get("subject", "").strip() or "Customer Support Enquiry"
    intent = state.get("intent", "")
    order_id = state.get("order_id", "")
    tool_result = state.get("tool_result", "")
    documents = state.get("retrieved_documents", [])

    knowledge_context = "\n\n".join(documents)

    # Handle missing order ID deterministically.
    if intent == "Order status enquiry" and not order_id:
        email = (
            f"Dear {name},\n\n"
            "Thank you for contacting us regarding your order.\n\n"
            "We would be happy to check its current status. "
            "Could you please provide your order ID so we can "
            "look into this for you?\n\n"
            "Thank you for your patience.\n\n"
            "Best regards,\n"
            "Customer Support Team"
        )

        return {"final_response": email}

    # Use verified tool output for known order IDs.
    if intent == "Order status enquiry" and order_id:
        if tool_result == "Order not found in the mock order database.":
            verified_status = (
                "We could not find this order in our current order "
                "records. Please verify the order ID or contact "
                "our support team for assistance."
            )

        else:
            verified_status = (
                f"We have checked our order status tool and "
                f"confirmed that order {order_id} is currently "
                f"{tool_result.lower()}."
            )

        email = (
            f"Dear {name},\n\n"
            f"Thank you for contacting us regarding the status "
            f"of your order {order_id}.\n\n"
            f"{verified_status}\n\n"
            "Please refer to your shipping confirmation email "
            "for tracking details, if available.\n\n"
            "If you have any further questions or need assistance, "
            "please don't hesitate to contact our support team. "
            "We are happy to help.\n\n"
            "Thank you for your patience and understanding.\n\n"
            "Best regards,\n"
            "Customer Support Team"
        )

        return {"final_response": email}

    # Use the language model for general policy questions.
    try:
        llm = load_llm()

        prompt = f"""
You are a professional customer-support email assistant.

Write a concise, polite, helpful email to the customer.

Customer name: {name}
Customer email address: {state.get("customer_email", "")}
Email subject: {subject}
Customer message:
{state.get("email_body", "")}

Identified intent:
{intent}

Retrieved support knowledge:
{knowledge_context}

Workflow plan:
{state.get("planning", "")}

Tool observation:
{state.get("observation", "")}

Rules:
1. Use the retrieved support knowledge for policy questions.
2. Do not invent refund timelines, cancellation eligibility,
   tracking numbers, customer IDs, or account details.
3. Do not claim that an order was cancelled or a refund was issued
   unless verified evidence confirms it.
4. If the knowledge base does not answer the question, explain
   politely that the support team must verify the information.
5. Do not claim to have performed an action that was not performed.
6. Start with "Dear {name},".
7. End with "Best regards, Customer Support Team".
8. Return only the email body. Do not add analysis or explanations.
"""

        response = llm.invoke(prompt)
        email = str(response.content).strip()

        if not email:
            raise ValueError("The language model returned an empty response.")

        # Remove an accidental subject heading if the model adds one.
        email = re.sub(
            r"^\s*Subject:.*?\n+",
            "",
            email,
            count=1,
            flags=re.IGNORECASE,
        )

        return {"final_response": email}

    except Exception as exc:
        fallback = (
            f"Dear {name},\n\n"
            "Thank you for contacting our customer support team "
            f"regarding your enquiry about '{subject}'.\n\n"
            "We have received your request. Our support team may "
            "need to verify the relevant information before giving "
            "you a confirmed answer.\n\n"
            "We appreciate your patience and understanding.\n\n"
            "Best regards,\n"
            "Customer Support Team"
        )

        return {
            "final_response": fallback,
            "error": (
                "The language model could not generate the email. "
                f"A safe fallback response was used. Details: {exc}"
            ),
        }


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_workflow():
    """Create the Experiment 7 workflow."""

    workflow = StateGraph(EmailState)

    workflow.add_node("rag_retrieval", rag_retrieval)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("action", action_node)
    workflow.add_node("observation", observation_node)
    workflow.add_node("final_response", final_response_node)

    workflow.set_entry_point("rag_retrieval")

    workflow.add_edge("rag_retrieval", "reasoning")
    workflow.add_edge("reasoning", "planning")
    workflow.add_edge("planning", "action")
    workflow.add_edge("action", "observation")
    workflow.add_edge("observation", "final_response")
    workflow.add_edge("final_response", END)

    return workflow.compile()


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.subheader("👤 Customer Information")

with st.form("customer_email_form"):
    customer_name = st.text_input(
        "Customer Name",
        value="Ravi",
    )

    customer_email = st.text_input(
        "Customer Email Address",
        value="ravi@example.com",
    )

    subject = st.text_input(
        "Email Subject",
        value="Order status enquiry",
    )

    email_body = st.text_area(
        "Customer Email",
        value="Hi, can you please tell me the status of my order ORD123?",
        height=150,
    )

    submitted = st.form_submit_button(
        "🚀 Run Experiment 7",
        use_container_width=True,
    )


if submitted:
    if not customer_name.strip():
        st.error("Please enter the customer's name.")

    elif not email_body.strip():
        st.error("Please enter the customer's email message.")

    else:
        initial_state: EmailState = {
            "customer_name": customer_name.strip(),
            "customer_email": customer_email.strip(),
            "subject": subject.strip(),
            "email_body": email_body.strip(),
            "order_id": "",
            "intent": "",
            "retrieved_documents": [],
            "reasoning": "",
            "planning": "",
            "action": "",
            "tool_result": "",
            "observation": "",
            "final_response": "",
            "error": "",
        }

        with st.spinner(
            "Running RAG retrieval and LangGraph workflow..."
        ):
            try:
                app = build_workflow()
                result = app.invoke(initial_state)

                st.session_state["experiment7_result"] = result

            except Exception as exc:
                st.error(f"Workflow execution failed: {exc}")


# ============================================================
# DISPLAY WORKFLOW RESULTS
# ============================================================

result = st.session_state.get("experiment7_result")

if result:
    st.success("Experiment 7 workflow completed!")

    st.divider()

    st.subheader("📚 Retrieved Knowledge")

    documents = result.get("retrieved_documents", [])

    if documents:
        for index, document in enumerate(documents, start=1):
            with st.expander(
                f"Retrieved Document {index}",
                expanded=True,
            ):
                st.write(document)
    else:
        st.warning(
            "No knowledge documents were retrieved. "
            "Check the knowledge-base file and application logs."
        )

    st.divider()

    st.subheader("🧠 Reasoning")
    st.text(result.get("reasoning", "No reasoning recorded."))

    st.divider()

    st.subheader("📋 Planning")
    st.write(result.get("planning", "No plan recorded."))

    st.divider()

    st.subheader("⚡ Action")
    st.code(
        result.get("action", "No action recorded."),
        language="text",
    )

    st.divider()

    st.subheader("👁️ Observation")
    st.text(
        result.get("observation", "No observation recorded.")
    )

    st.divider()

    st.subheader("📨 Final Response")

    final_email = result.get("final_response", "").strip()

    if final_email:
        st.text_area(
            "Generated Customer Email",
            value=final_email,
            height=300,
            key="generated_email_display",
        )

        st.download_button(
            label="⬇️ Download Customer Email",
            data=final_email,
            file_name="customer_support_email.txt",
            mime="text/plain",
        )
    else:
        st.error(
            "The workflow completed, but the final response is empty."
        )

    if result.get("error"):
        st.warning(result["error"])

    st.divider()

# ============================================================
# EXPLANATION OF RAG
# ============================================================

with st.expander("ℹ️ How does RAG work in this project?", expanded=False):
    st.markdown(
        """
        **1. Load the knowledge document**  
        Read customer-support policies from the knowledge-base file.

        **2. Split the document**  
        Divide the text into smaller, overlapping chunks.

        **3. Generate embeddings**  
        Convert the chunks into numerical vectors using the
        Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model.

        **4. Store embeddings**  
        Store the vectors in a FAISS index for similarity search.

        **5. Retrieve relevant knowledge**  
        Retrieve the three most relevant text chunks for the
        customer's email.

        **6. Reasoning**  
        Identify the customer's intent and determine which
        information is needed.

        **7. Planning**  
        Choose the appropriate next action.

        **8. Action**  
        Execute the order-status lookup when an order ID is
        available and the request concerns order status.

        **9. Observation**  
        Display the actual tool result and retrieved knowledge.

        **10. Final response**  
        Generate a professional email using verified information
        and the retrieved support policies.
        """
    )

st.caption(
    "Optimization: The embedding model, FAISS index, and Ollama "
    "model are cached with st.cache_resource."
)