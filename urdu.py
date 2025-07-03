import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
import streamlit as st

# Configs
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
genai.configure(api_key="******************************")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Step 1: Extract images
def extract_images_from_pdf(pdf_bytes):
    images = []
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(pdf_doc)):
        pix = pdf_doc[page_num].get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)
    return images

# Step 2: OCR
def ocr_images(images):
    urdu_text = ""
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="urd")
        urdu_text += f"\n\n--- صفحہ {i+1} ---\n\n{text}"
    return urdu_text

# Step 3: Clean text
def clean_text(text):
    cleaned = text.replace("\x0c", "")
    cleaned = "\n".join([line.strip() for line in cleaned.split("\n") if line.strip()])
    return cleaned

# Step 4: Split, embed, and store
def split_and_embed(text):
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "۔", ".", "!", "؟", "  "]
    )
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")

    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embedding,
        url="*********************************************************************************",
        api_key="*****************************************************************************",
        collection_name="urdu"
    )

    return vectorstore, chunks

# Step 5: Ask Gemini
def ask_gemini(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""تمھیں اردو متن کے بارے میں سوالات کے جوابات دینے ہیں۔

سیاق و سباق:
{context}

سوال:
{query}

جواب:"""
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="📚 Urdu RAG Chatbot", layout="centered")
st.title("📘 Urdu PDF Q&A with Gemini + Qdrant")

uploaded_pdf = st.file_uploader("Upload an Urdu PDF file", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Processing PDF..."):
        images = extract_images_from_pdf(uploaded_pdf.read())

    st.success(f"✅ Extracted {len(images)} page(s) from PDF.")

    raw_text = ocr_images(images)
    if not raw_text.strip():
        st.error("❌ No text extracted via OCR. Check the image quality.")
    else:
        cleaned_text = clean_text(raw_text)
        vectorstore, chunks = split_and_embed(cleaned_text)
        st.success(f"✅ Split into {len(chunks)} chunks and stored in Qdrant.")

        user_question = st.text_input("📨 Ask a question in Urdu:")

        if user_question:
            with st.spinner("🔎 Retrieving answer..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(user_question)
                answer = ask_gemini(user_question, relevant_docs)

            st.markdown("### 📘 Urdu Response:")
            st.write(answer)
