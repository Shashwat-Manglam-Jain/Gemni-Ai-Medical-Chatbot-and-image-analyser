from flask import Flask, request, render_template, jsonify
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from textblob import TextBlob
from PIL import Image, ImageEnhance, ImageFilter
import google.generativeai as genai
import easyocr
import pytesseract
import numpy as np
import re
import io
import os
import logging

# ===========================
#  Logging Setup
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===========================
#  Gemini API Setup
# ===========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ===========================
#  Vector Store Setup
# ===========================
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    vs = FAISS.load_local("faiss_index1", emb, allow_dangerous_deserialization=True)
except Exception as e:
    logger.error(f"Failed to load FAISS index: {e}")
    raise

# new retriever API still works the same, but the method changed
retriever = vs.as_retriever(search_kwargs={"k": 5})

# ===========================
#  Helper Functions
# ===========================
def truncate_to_token_limit(text, max_tokens=512):
    """Rough token limiter."""
    return text[: max_tokens * 4]

def correct_typos(text):
    try:
        corrected = str(TextBlob(text).correct())
        medical_terms = {
            "mg", "ml", "tablet", "caplet", "capsule", "dose", "vicodin",
            "acetaminophen", "hrs", "prescription", "nonprescription",
            "pharmacist", "drowsiness", "constipation", "liver",
            "allergic", "mfg"
        }
        for term in medical_terms:
            if term in text.lower() and term not in corrected.lower():
                corrected = corrected.replace(term.lower(), term)
        fixes = {
            "camlet": "caplet", "thrs": "this", "violin": "vicodin",
            "his": "hrs", "my": "mg", "river": "liver",
            "alkergic lo": "allergic to", "pharmaclst": "pharmacist",
            "phamacist": "pharmacist", "nonprescriptionj": "nonprescription",
            "nnprescriptionj": "nonprescription", "am ": "do ",
            "mug": "mfg", "acelaminophen": "acetaminophen", "clay": "cla",
        }
        for k, v in fixes.items():
            corrected = corrected.replace(k, v)
        return corrected
    except Exception as e:
        logger.warning(f"Typo correction failed: {e}")
        return text

def preprocess_query(query):
    query = query.strip().lower()
    query = re.sub(r"[^\w\s\d/-]", "", query)
    return correct_typos(query)

def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        img = img.resize((int(img.width * 2), int(img.height * 2)), Image.LANCZOS)
        gray = img.convert("L")
        contrast = ImageEnhance.Contrast(gray).enhance(1.5)
        sharp = contrast.filter(ImageFilter.SHARPEN)
        denoised = sharp.filter(ImageFilter.MedianFilter(size=1))
        thresh = denoised.point(lambda p: 255 if p > 120 else 0)
        thresh.save("preprocessed.png")
        return thresh
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def extract_medications(text: str):
    meds, dosages = [], []
    med_pat = re.compile(r"\b(vicodin\s*es|acetaminophen\s*\d+\s*mg)\b", re.I)
    meds.extend([m.strip() for m in med_pat.findall(text) if m.strip()])
    dose_pat = re.compile(
        r"(?:take|by\s*mouth)\s*(\d+)\s*(?:tablet|caplet)\s*(?:every|as\s*needed)?\s*(\d+-\d+\s*hrs)?"
        r"(?:\s*or\s*as\s*needed)?(?:\s*no\s*more\s*than\s*(\d+)\s*(?:tablet|caplet)s?\s*in\s*24\s*hrs)?",
        re.I,
    )
    found_doses = dose_pat.findall(text)
    if found_doses:
        tablet_count, schedule, max_dose = found_doses[0]
        dosage_info = f"Take {tablet_count} tablet(s) {schedule or 'as needed'}"
        if max_dose:
            dosage_info += f", no more than {max_dose} tablets in 24 hrs"
        dosages.append(dosage_info.strip())
    return {"medications": meds, "dosages": dosages} if meds or dosages else {}

# ===========================
#  OCR Setup
# ===========================
reader = easyocr.Reader(["en"], gpu=os.environ.get("CUDA_AVAILABLE", "False") == "True")

# ===========================
#  Flask Routes
# ===========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_sources", methods=["POST"])
def get_sources():
    try:
        q = preprocess_query(request.form.get("question", ""))
        if not q:
            return jsonify({"error": "No question provided"}), 400

      
        docs = retriever.invoke(q)
        chunks = [d.page_content for d in docs]
        return jsonify({"sources": chunks})
    except Exception as e:
        logger.error(f"Error in get_sources: {e}")
        return jsonify({"error": "Failed to retrieve sources"}), 500

@app.route("/get_answer", methods=["POST"])
def get_answer():
    try:
        question = request.form.get("question", "")
        if not question:
            return jsonify({"answer": "No question provided"}), 400

        query = preprocess_query(question)
        docs = retriever.invoke(query)  
        context = " ".join([truncate_to_token_limit(d.page_content, 512) for d in docs])
        prompt = f"""
You are a factual, concise medical assistant. 
Use the provided medical context to generate a clear, structured answer that focuses on **treatment and medications** for the disease or condition asked about. 

Instructions:
- Emphasize **specific drugs or therapeutic regimens** if mentioned (e.g., artemisinin, chloroquine, primaquine).
- If medications are not mentioned in the context, respond with: 
  “The text does not specify any drug treatment, but here is the general management approach based on the provided information.”
- Do NOT invent or guess any medication that is not in the context.
- Write your answer in a professional, concise, and bulleted format.
- Start with: “To treat {query}, the text suggests:”

<context>
{context}
</context>

Question: {query}
"""



        prompt = truncate_to_token_limit(prompt, 512)

        gemini_response = gemini_model.generate_content(prompt)
        return jsonify({"answer": gemini_response.text})

    except Exception as e:
        logger.exception("Error in /get_answer")
        return jsonify({"error": f"Error: {e}"}), 500

@app.route("/analyze_prescription", methods=["POST"])
def analyze_prescription():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        proc_img = preprocess_image(img)
        img_np = np.array(proc_img)

        lines = reader.readtext(img_np, detail=0, paragraph=True, allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-# ")
        text = "\n".join(lines).strip()

        if not text or len(text.split()) < 5:
            text = pytesseract.image_to_string(proc_img, config="--psm 6 --oem 3").strip()

        if not text:
            return jsonify({
                "error": "No text extracted from image.",
                "extracted_text": "",
                "analysis": "",
                "medications": [],
                "dosages": []
            }), 200

        # Clean up text and analyze
        clean_text = preprocess_query(text)
        meds_info = extract_medications(clean_text)

        patient_info = {
            "name": "Unknown",
            "date": "N/A",
            "dispensed_by": "Unknown"
        }

        prompt = (
            "You are a medical assistant. Analyze this prescription. "
            "Extract patient info, medications, dosage, and provide a short summary.\n\n"
            f"Prescription text:\n{clean_text}"
        )

        gemini_output = gemini_model.generate_content(prompt)

        return jsonify({
            "extracted_text": clean_text,
            "patient_info": patient_info,
            "medications": meds_info.get("medications", []),
            "dosages": meds_info.get("dosages", []),
            "analysis": gemini_output.text or "No AI summary generated."
        })

    except Exception as e:
        logger.exception("Error in analyze_prescription")
        return jsonify({"error": f"An error occurred while analyzing the prescription: {str(e)}"}), 500

# ===========================
#  Run the App
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

