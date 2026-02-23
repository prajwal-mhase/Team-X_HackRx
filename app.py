import streamlit as st
import PyPDF2
import json
import re
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.types import GenerateContentConfig
from secret import GEMINI_API_KEY

# ✅ Initialize Gemini Client (Modern SDK)
client = genai.Client(api_key=GEMINI_API_KEY)

MAX_CHARS_PER_CHUNK = 20000


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


# ✅ Chunk text into safe pieces
def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    words = text.split()
    chunks, current_chunk = [], ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


# ✅ Extract JSON from response
def extract_json_from_text(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found.")
    return json.loads(match.group(0))


# ✅ Analyze chunk
def analyze_chunk(query, chunk_text, chunk_index, total_chunks):
    prompt = f"""
You are an AI insurance analyst.

User Query:
{query}

Policy Document (Part {chunk_index+1} of {total_chunks}):
{chunk_text}

Identify relevant clauses and exclusions.
Do NOT make final decision.
Summarize findings only.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0)
    )

    return f"--- Findings from Part {chunk_index+1} ---\n{response.text}"


# ✅ Final decision
def ask_llm(query, merged_summary):
    prompt = f"""
You are an AI insurance analyst.

User Query:
{query}

Summarized Relevant Policy Information:
{merged_summary}

Return ONLY JSON:

{{
  "decision": "approved or rejected",
  "amount": "amount in INR or null",
  "justification": "your reasoning"
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(temperature=0)
    )

    return response.text


# ✅ Main UI
def main():
    st.set_page_config(page_title="Insurance Claim Analyzer", page_icon="📄", layout="wide")

    with st.sidebar:
        st.title("Claim Analyzer")
        st.markdown("**🔐 AI-powered system to verify insurance claims.**")
        st.info("Upload your insurance policy and describe your claim in simple terms.")
        st.caption("Built with Streamlit • Gemini • PyPDF2")

    st.markdown("<h2 style='color:#004080'>📑 Insurance Claim Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("Use this tool to automatically verify claims based on uploaded insurance documents.")

    uploaded_file = st.file_uploader("📤 Upload Insurance Policy PDF", type="pdf")

    query = st.text_area(
        "✍️ Describe your claim below",
        placeholder="Example: 46-year-old male, knee surgery in Pune, 3-month-old insurance policy",
        height=100
    )

    if uploaded_file and query:

        with st.spinner("📚 Extracting policy content..."):
            doc_text = extract_pdf_text(uploaded_file)

        if not doc_text:
            st.error("❌ No readable text found.")
            return

        if st.button("🚀 Analyze Claim Now"):

            chunks = chunk_text(doc_text)
            st.info(f"Document split into {len(chunks)} chunk(s).")

            partial_findings = []

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(analyze_chunk, query, chunk, idx, len(chunks))
                    for idx, chunk in enumerate(chunks)
                ]
                for future in as_completed(futures):
                    partial_findings.append(future.result())

            partial_findings.sort(key=lambda x: int(re.search(r"Part (\d+)", x).group(1)))
            merged_summary = "\n\n".join(partial_findings)

            try:
                raw = ask_llm(query, merged_summary)
                result = extract_json_from_text(raw)

                st.success("✅ Claim Analysis Complete")

                decision = result.get("decision", "").lower()
                amount = result.get("amount")
                justification = result.get("justification", "")

                if decision == "approved":
                    st.success("✅ Claim Approved")
                elif decision == "rejected":
                    st.error("❌ Claim Rejected")
                else:
                    st.warning("⚠️ Unclear decision")

                if amount and amount != "null":
                    try:
                        st.metric("💰 Amount Payable", f"₹{int(amount):,}")
                    except:
                        st.metric("💰 Amount Payable", amount)

                st.markdown("### 🧠 Justification")
                st.write(justification)

                with st.expander("📦 Full Gemini JSON Output"):
                    st.json(result)

            except Exception as e:
                st.error("⚠️ Error processing Gemini response")
                st.exception(e)

    else:
        st.warning("📥 Please upload a PDF and describe your claim to start.")


if __name__ == "__main__":
    main()