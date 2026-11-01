# 📧 AI-Powered Email Auto-Responder Using LangChain and Ollama

An AI-powered Email Assistant built using **Python, Streamlit, LangChain, Ollama, and Large Language Models (LLMs)**. This project explores how AI agents can understand incoming emails, classify customer requests, determine their priority, select appropriate actions, and generate email responses.

The project is organized into seven experiments, each exploring different approaches to email automation, from basic workflow demonstrations to more advanced agent-based processing.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Technologies Used](#-technologies-used)
- [Project Experiments](#-project-experiments)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [Installation and Setup](#-installation-and-setup)
- [Running the Experiments](#-running-the-experiments)
- [Sample Input](#-sample-input)
- [Example Output](#-example-output)
- [Workflow](#-workflow)
- [Key Features](#-key-features)
- [Future Enhancements](#-future-enhancements)
- [Important Notes](#-important-notes)

---

## 🚀 Project Overview

Customer support teams frequently receive emails related to order status, cancellations, refunds, product inquiries, and other service requests. Manually processing every email can be time-consuming.

This project explores an AI-based approach to automating email processing using Large Language Models and AI agents.

The application accepts customer email details, processes the message, identifies the customer's intent, determines its priority, selects an appropriate action, and generates a response when supported by the experiment's implementation.

The experiments progressively explore different email-processing architectures using LangChain and locally running language models through Ollama.

---

## 🎯 Objectives

- Automate basic email-processing workflows.
- Identify the intent of incoming customer emails.
- Classify emails according to priority.
- Determine whether a response is required.
- Generate structured email-classification results using JSON.
- Explore multi-agent architectures for email processing.
- Generate customer-support email drafts.
- Build interactive interfaces using Streamlit.
- Experiment with locally running LLMs using Ollama.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application interface |
| LangChain | Framework for working with language models and prompts |
| Ollama | Runs language models locally |
| Llama 3.2 | Language model used in the experiments |
| LangGraph | Supports graph-based agent workflows where implemented |
| FAISS | Vector similarity search for retrieval-based applications where implemented |
| Hugging Face Sentence Transformers | Text embeddings for retrieval-based applications where implemented |
| JSON | Structured representation of classification results |
| Git and GitHub | Version control and project hosting |

---

## 🧪 Project Experiments

The project contains seven experimental implementations.

### Experiment 1: Basic Email Assistant

**Objective:** Demonstrate the basic email-processing workflow.

Features:
- Receive customer email details.
- Demonstrate email understanding and intent identification.
- Display knowledge-base search stages.
- Present a basic acknowledgement.
- Display workflow completion.

### Experiment 2: Single-Agent Email Assistant

**Objective:** Process emails using a single AI agent powered by LangChain and Ollama.

Features:
- Understand incoming emails.
- Identify customer intent.
- Determine email priority.
- Select the required action.
- Generate a customer-support email response.

### Experiment 3: Email Classification and Structured Output

**Objective:** Classify emails and represent the results in a structured format.

Features:
- Identify email intent.
- Determine priority.
- Decide whether a response is required.
- Recommend an appropriate action.
- Generate structured JSON output.
- Display the classification results.

Example classification:

```json
{
  "intent": "Request for Order Cancellation",
  "priority": "Medium",
  "response_required": true,
  "action": "Respond to customer and provide cancellation instructions"
}
```

### Experiment 4: Multi-Agent Email System

**Objective:** Explore a multi-agent architecture for processing customer emails.

The workflow uses three logical agents:

**Agent 1 — Email Filter**
- Determines whether an email is important.
- Provides a reason for the classification.

**Agent 2 — Email Action**
- Determines the action required.
- Assigns a priority to the email.

**Agent 3 — Response Writer**
- Generates an email subject.
- Drafts a customer-support response.

The application combines the agent results into a final email draft.

### Experiment 5: Advanced Email Processing

**Objective:** Explore additional email-processing functionality.

This experiment is part of the project's experimental series. Its specific capabilities depend on the implementation in `app5.py`.

### Experiment 6: Advanced Email Assistant

**Objective:** Continue experimenting with AI-powered email automation.

The project includes `app6.py` and `app6_streamlit.py`. Refer to these files for the exact functionality implemented in Experiment 6.

### Experiment 7: RAG-Based Email Assistant

**Objective:** Explore retrieval-augmented email processing.

The project includes `app7.py`, which was successfully launched during development.

The RAG approach can combine information retrieved from a knowledge base with language-model generation to produce context-aware responses.

The specific retrieval and response-generation behavior depends on the implementation and configured knowledge base.

---

## 📂 Project Structure

```text
Email-Auto-Responder/
│
├── app1.py
├── app2.py
├── app3.py
├── app4.py
├── app5.py
├── app6.py
├── app6_streamlit.py
├── app7.py
│
├── knowledge_base/
│
├── test_langchain.py
├── requirements.txt
├── .gitignore
│
└── README.md
```

**Note:** The project directory may contain additional files as development continues. Keep the structure in this README synchronized with your actual GitHub repository.

---

## 💻 System Requirements

Before running the project, ensure that you have installed:

- Python 3.10 or later, compatible with the installed dependencies.
- Ollama.
- Git.
- A modern web browser.
- Sufficient memory and processing resources to run the selected language model locally.

The project was developed and tested in a Windows environment using Python 3.13.

---

## ⚙️ Installation and Setup

### Step 1: Install Ollama

Download and install Ollama from its official website:

https://ollama.com/

### Step 2: Download the language model

Open PowerShell or a terminal and run:

```bash
ollama pull llama3.2
```

To test the model:

```bash
ollama run llama3.2
```

You can exit the interactive session after confirming that the model responds.

### Step 3: Clone the repository

Replace `<your-github-username>` with your GitHub username.

```bash
git clone https://github.com/<your-github-username>/RAG-Email-Auto-Responder.git
```

Navigate to the project directory:

```bash
cd RAG-Email-Auto-Responder
```

### Step 4: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
```

Install the project requirements:

```bash
pip install -r requirements.txt
```

### Step 6: Verify Ollama

Make sure Ollama is installed and the model is available:

```bash
ollama list
```

If the model is not listed, download it using:

```bash
ollama pull llama3.2
```

---

## ▶️ Running the Experiments

Run the desired experiment from your project directory.

### Experiment 1

```bash
streamlit run app1.py
```

### Experiment 2

```bash
streamlit run app2.py
```

### Experiment 3

```bash
streamlit run app3.py
```

### Experiment 4

```bash
streamlit run app4.py
```

### Experiment 5

```bash
streamlit run app5.py
```

### Experiment 6

```bash
streamlit run app6.py
```

If Experiment 6 uses the dedicated Streamlit implementation, run:

```bash
streamlit run app6_streamlit.py
```

### Experiment 7

```bash
streamlit run app7.py
```

After launching an application, open the local URL displayed in your terminal. Streamlit commonly uses:

```text
http://localhost:8501
```

If multiple applications are already running, Streamlit may select another available port.

**Important:** These commands assume that each corresponding file contains a Streamlit application. If an experiment uses a different entry point, follow the instructions in its source code.

---

## 📨 Sample Input

Use the following sample email to test the application, where the corresponding experiment supports these fields.

**Sender Name:** Rahul

**Sender Email:** rahul@example.com

**Email Subject:** Order Status Inquiry

**Email Message:**

```text
Hello Customer Support,

I recently placed an order and would like to know its
current status.

My order number is ORD123.

Could you please provide an update?

Thank you,
Rahul
```

---

## 📊 Example Output

The following is an illustrative example of a structured classification result.

```json
{
  "intent": "Order Status Inquiry",
  "priority": "Medium",
  "response_required": true,
  "action": "Check order status and respond to the customer"
}
```

A generated email draft could look like this:

```text
Subject: Re: Order Status Inquiry

Dear Rahul,

Thank you for contacting customer support regarding
your order.

We will check the current status of your order and
provide you with an update.

Best regards,
Customer Support
```

**Note:** This is an illustrative example, not a guaranteed application output. Order status and delivery information should be retrieved from a reliable data source before being presented as confirmed facts.

---

## 🔄 General Email Processing Workflow

```text
        Customer Email
              |
              v
       Email Reception
              |
              v
      Email Understanding
              |
              v
       Intent Detection
              |
              v
      Priority Assessment
              |
              v
       Action Selection
              |
              v
     Knowledge Retrieval
       (if implemented)
              |
              v
      Response Generation
              |
              v
       Final Email Draft
```

The exact workflow differs between experiments. For example, Experiment 4 explores separate agents for filtering, action selection, and response writing.

---

## ✨ Key Features

- **AI-powered email processing:** Uses language models to understand customer messages.
- **Intent classification:** Identifies the purpose of an email.
- **Priority assessment:** Assigns priority levels to customer requests.
- **Action selection:** Determines an appropriate next step.
- **Structured output:** Represents classification results using JSON.
- **Multi-agent architecture:** Separates email-processing responsibilities across agents.
- **Email drafting:** Generates customer-support response drafts.
- **Local model execution:** Uses Ollama to run the selected language model locally.
- **Interactive interface:** Provides a Streamlit-based interface for testing experiments.
- **Extensible architecture:** Supports further experimentation with retrieval and agent workflows.

---

## 🔮 Future Enhancements

Potential improvements include:

- Integrating Gmail or Outlook for real email ingestion.
- Connecting to an actual order-management database.
- Implementing reliable refund and cancellation verification.
- Adding a retrieval-augmented generation pipeline with a maintained knowledge base.
- Using LangGraph for explicit multi-step workflows where appropriate.
- Adding human approval before sending generated emails.
- Introducing confidence scores for email classification.
- Improving error handling, logging, and retry mechanisms.
- Adding unit tests and automated evaluation of generated responses.
- Measuring classification accuracy and response-generation quality.
- Deploying the application to a cloud platform.
- Adding authentication and role-based access control.

---

## 🔐 Important Notes

1. **Local execution:** Ollama runs the selected model locally. This does not automatically guarantee that every component of the application operates offline; dependencies and model downloads may require internet access.

2. **Generated responses:** Language models can generate inaccurate information. Verify important facts against trusted sources.

3. **Order and refund information:** Do not claim that an order has shipped or a refund has been processed unless the information has been verified.

4. **Email sending:** Generating an email draft is different from sending an email. Configure an email provider and appropriate authorization if actual sending functionality is required.

5. **Privacy:** Avoid including real customer information, passwords, API keys, or email credentials in the repository.

6. **Environment files:** Keep `.venv/`, `.env`, and other sensitive configuration files out of version control.

7. **Dependencies:** The exact packages required may differ between experiments. Update `requirements.txt` whenever the implementation changes.

---

## 👨‍💻 Author

**Srineeth Reddy**

GitHub: https://github.com/Srineeth19

---

## 📄 License

This project is intended for educational and experimental purposes.

If you plan to distribute or publish the project, consider adding a `LICENSE` file that specifies how others may use, modify, and distribute your code.
