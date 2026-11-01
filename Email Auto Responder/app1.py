import streamlit as st

# ---------------------------------------
# Email Assistant - Experiment 1
# ---------------------------------------

st.title("📧 Email Assistant")

st.write(
    "Welcome to the Email Auto-Responder - Experiment 1"
)

# ---------------------------------------
# User Input
# ---------------------------------------

name = st.text_input("Enter Your Name")

email = st.text_input("Enter Your Email Address")

subject = st.text_input("Enter Email Subject")

message = st.text_area("Enter Your Email Message")

# ---------------------------------------
# Process Email
# ---------------------------------------

if st.button("Process Email"):

    st.subheader("Agent Workflow")

    st.write("Step 1 : Email Received")

    st.write("Step 2 : Understanding Email")

    st.write("Step 3 : Identifying Email Intent")

    # Decision Point
    if "order" in message.lower() or "status" in message.lower():

        st.write("Step 4 : Intent Identified - Order Information")

        st.write("Step 5 : Searching Knowledge Base")

        st.write("Step 6 : Information Found")

        st.success(
            "Thank you for contacting us. "
            "Your request has been received."
        )

        st.write("Step 7 : Response Generated")

    else:

        st.write("Step 4 : Intent Not Recognized")

        st.warning(
            "The Email Assistant could not understand "
            "the request."
        )

    st.write("Step 8 : Email Processing Completed")