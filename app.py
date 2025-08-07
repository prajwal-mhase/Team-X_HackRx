import streamlit as st
import PyPDF2
import json
import re
from io import BytesIO
from secret import GEMINI_API_KEY
import google.generativeai as genai

# ✅ Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# ✅ Extract JSON from LLM response
def extract_json_from_text(text):
    try:
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No JSON object found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

# ✅ Query Gemini with structured prompt
def ask_llm(query, document_text):
    prompt = f"""
You are an AI insurance analyst.

User Query:
{query}

Policy Document:
{document_text}

Your task:
1. Determine if the claim should be approved or rejected.
2. Specify the claim amount (if applicable).
3. Justify the decision.

Return ONLY a JSON response in the following format (no markdown, no explanation):

{{
  "decision": "approved or rejected",
  "amount": "amount in INR or null",
  "justification": "your reasoning"
}}
"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.0})
    return response.text.strip()

# ✅ Main UI
def main():
    st.set_page_config(page_title="Insurance Claim Analyzer", page_icon="📄", layout="wide")

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("Claim Analyzer")
        st.markdown("**🔐 AI-powered system to verify insurance claims.**")
        st.info("Upload your insurance policy and describe your claim in simple terms.")
        st.caption("Built with Streamlit • Gemini • PyPDF2")

    st.markdown("<h2 style='color:#004080'>📑 Insurance Claim Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("Use this tool to **automatically verify claims** based on uploaded insurance documents.")

    with st.expander("🧭 How to Use", expanded=False):
        st.markdown("""
        1. Upload an **insurance policy** PDF 📄  
        2. Enter your claim as a **natural language description** 💬  
        3. Click **Analyze** to get:  
            - ✅ Decision (Approved or Rejected)  
            - 💰 Estimated Amount  
            - 🧠 Explanation from Gemini
        """)

    uploaded_file = st.file_uploader("📤 Upload Insurance Policy PDF", type="pdf")
    default_query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    query = st.text_area("✍️ Describe your claim below", value=default_query, height=100)

    if uploaded_file and query:
        with st.spinner("📚 Extracting policy content..."):
            doc_text = extract_pdf_text(uploaded_file)

        if not doc_text:
            st.error("❌ No readable text found in the uploaded document.")
            return

        if st.button("🚀 Analyze Claim Now"):
            with st.spinner("🤖 Processing claim with Gemini..."):
                try:
                    raw = ask_llm(query, doc_text)
                    result = extract_json_from_text(raw)

                    st.success("✅ Claim Analysis Complete")
                    st.markdown("### 📊 Result Summary")
                    col1, col2 = st.columns(2)

                    with col1:
                        decision = result.get("decision", "").lower()
                        if decision == "approved":
                            st.success("✅ Claim Approved")
                        elif decision == "rejected":
                            st.error("❌ Claim Rejected")
                        else:
                            st.warning("⚠️ Unclear decision")

                        amt = result.get("amount")
                        st.metric("💰 Amount Payable", f"₹{int(amt):,}" if amt else "Not specified")

                    with col2:
                        st.markdown("#### 🧠 Justification")
                        st.markdown(result.get("justification", "No justification provided."))

                    with st.expander("📦 Full Gemini JSON Output"):
                        st.json(result)

                except Exception as e:
                    st.error("⚠️ Error processing Gemini response")
                    st.text(raw if 'raw' in locals() else "No response received.")
                    st.exception(e)
    else:
        st.warning("📥 Please upload a PDF and describe your claim to start.")

if __name__ == "__main__":
    main()
