from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from openai import OpenAI
import tiktoken
import json
import re
import fitz
import os
import hmac
import hashlib
from supabase_client import fetch_supabase_db, fetch_supabase_cat_db
from pydantic import BaseModel
import requests
import asyncio


def safe_parse_json(raw):
    """
    Try to parse JSON, fix minor truncation issues if possible.
    """
    raw = raw.strip()
    # Remove ```json or ``` markdown if present
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"```$", "", raw)

    # Try normal parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if raw.startswith("[") and not raw.endswith("]"):
            raw += "]"
        try:
            return json.loads(raw)
        except:
            return None

# --- Helper to chunk list ---
def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

# --- Count tokens function ---
def count_tokens(prompt, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


# --- Safe JSON parser (reuse) ---
def safe_parse_json(raw):
    import re, json
    raw = raw.strip()
    # Remove ```json or ``` markdown if present
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"```$", "", raw)

    # Try normal parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Quick fix: ensure ending ]
        if raw.startswith("[") and not raw.endswith("]"):
            raw += "]"
        try:
            return json.loads(raw)
        except:
            return None

## new prompt

# --- Helper to chunk list ---
def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

# --- Count tokens function ---
def count_tokens(prompt, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(prompt))


# --- Safe JSON parser (reuse) ---
def safe_parse_json(raw):
    import re, json
    raw = raw.strip()
    # Remove ```json or ``` markdown if present
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"```$", "", raw)

    # Try normal parsing
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Quick fix: ensure ending ]
        if raw.startswith("[") and not raw.endswith("]"):
            raw += "]"
        try:
            return json.loads(raw)
        except:
            return None

# --- Batch categorization function with confidence and date ---
def categorize_transactions_batch(client, df, amount_threshold=100, batch_size=20, model="gpt-4o-mini", person_name='Abhishek', mobile_numbers = '7206527787'):
    """
    Categorize transactions using LLM in batches with confidence scores.
    Handles 'Narration', 'Credit Amount', 'Debit Amount', and 'Date' columns.
    Returns df with new 'Amount', 'Category', 'Confidence' columns.
    """
    results = []

    # Unified Amount column
    df["Amount"] = df["Credit Amount"].fillna(0) - df["Debit Amount"].fillna(0)
    df.rename(columns={"Narration": "Description"}, inplace=True)

    # Filter rows for LLM
    df_to_process = df[df["Amount"].abs() >= amount_threshold].copy()

    for batch in chunker(list(df_to_process.itertuples(index=False)), batch_size):
        # Prepare transactions text including Date
        transactions_text = "\n".join(
            [f"- Description: {row.Description}, Amount: {row.Amount}, Date: {row.Date}" for row in batch]
        )

        # Prompt with confidence and date
        prompt = f"""
        You are an expert accounting assistant. Categorize the following bank transactions using Description, Amount, and Date.
        
        Rules for categorization:
        1. Categories include (but are not limited to): 
           Food, Travel, Office Supplies, Client Expense, Dividend, Investment, Investment Withdrawal, 
           Profit from Investment, Interest, Tax, Regular recurring Salary, UPI Transfer, Shopping, 
           Entertainment, Rent, Self transfer, Mobile Recharge, Miscellaneous.
        
        2. Specific clarifications:
           - If Amount > 0 (deposit/credit), it CANNOT be categorized as "Investment".
             • If it is a return/profit/redemption from investment, use "Profit from Investment" or "Investment Withdrawal".
           - If Description contains "ACH D- INDIAN CLEARING CORP" or similar clearing corp, categorize as "Investment".
           - If Description contains "ZEPTO", "ZOMATO", "SWIGGY", or other food merchants, categorize as "Food".
           - For UPI transactions:
             • If it matches known merchants, map accordingly.
             • If it is clearly P2P (names/emails/phone numbers), categorize as "UPI Transfer".
             • If the transaction description or UPI ID contains the **account holder name or mobile number(s)** provided to you (see *Person data* below), treat it as a "Self transfer" when sender and receiver are the same person.
           - Salary credits should be mapped to "Regular recurring Salary".
        
        3. **Consistency rule (NEW)**:
           - If multiple transactions in this batch share the same normalized merchant name (normalize by removing numeric IDs, UTR strings, `CR/DR` tokens, and punctuation), ensure they are assigned the **same Category** across the batch. If the model is unsure, choose the category that appears most frequent among those similar transactions (majority). If a tie, choose the category with higher confidence.
        
        4. **Person data (INPUT; NEW)**:
           - Person name (from function): `{person_name}`
           - Mobile numbers (from function): `{mobile_numbers}`
           - Use these to identify "Self transfer" — i.e., if the narration contains the same person name or any of the mobile numbers, prefer "Self transfer" when the flow looks like an internal move.
        
        5. **Confidence labels (NEW)**:
           - Instead of numeric confidence, return one of: `"very high"`, `"high"`, `"medium"`, `"low"`.
           - Use `"very high"` when a deterministic rule or exact merchant match applies (e.g., merchant in known list).
           - Use `"high"` for strong semantic matches or clear patterns.
           - Use `"medium"` for plausible guesses.
           - Use `"low"` if you are unsure or the narration is ambiguous.

        Output format:
        Respond ONLY in valid JSON format as a list of objects with keys:
        Description, Amount, Date, Category, Confidence, Reason
        
        Notes:
        - Reason must be 1–3 words only, explaining why the Category was chosen.
          Example: "Zepto", "salary credit Paytm", "p2p Vijay", "investment corp ACH", "investment corp Zerodha" etc.
        
        Transactions:
        {transactions_text}
        """

        # Optional: print token length
        print("🔹 Prompt token length:", count_tokens(prompt, model="gpt-4o-mini"))

        # LLM call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15000,
        )

        # Parse JSON output
        raw_output = response.choices[0].message.content
        batch_results = safe_parse_json(raw_output)
        if batch_results:
            results.extend(batch_results)
        else:
            print("⚠️ Failed to parse JSON for batch. Raw output:", raw_output)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def extract_text(doc):
    all_lines = []
    for page in doc:
        text = page.get_text("text")
        lines = text.split('\n')
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        all_lines.extend(lines)
    extracted_text = "\n".join(all_lines)
    return extracted_text

def extract_text_from_url(pdf_url):
    # Download PDF to temp file
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {response.status_code}")

    pdf_bytes = response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return extract_text(doc)
        
def pdf_to_csv(extracted_text, client, model):

    # Construct the prompt
    prompt = f"""
    You are an AI assistant specialized in parsing bank statements. 
        Extract all transactions into CSV with relevant columns: Date,Narration,Debit Amount, Credit Amount
        Combine multi-line narrations. Ignore headers/footers.
        Output the result **only** as CSV. Do not add explanations, quotes, or Markdown.
        Here is the extracted text:
        {extracted_text}
    """
    
    print("🔹 Prompt token length:", count_tokens(prompt, model="gpt-4o-mini"))
    
    # Make the OpenAI API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=15000  
    )
    
    # Extract the CSV from the response
    csv_output = response.choices[0].message.content
    # Save to CSV
    with open("bank_statement_parsed.csv", "w", encoding="utf-8") as f:
        f.write(csv_output)
    df = pd.read_csv("bank_statement_parsed.csv")
    
    return df
    
def df_to_event_list(df, client_id, file_id, accountant_id):
    # Select required columns and convert to list of dicts
    required_cols = ['category', 'confidence', 'reason']
    event_list = df[required_cols].to_dict(orient='records')

    cat_id_map = fetch_supabase_cat_db()
    name_to_id_map = {item['name']: item['id'] for item in cat_id_map}
    # Add additional info to each event object
    for event in event_list:
        event['client_id'] = client_id
        event['file_id'] = file_id
        event['accountant_id'] = accountant_id
        event['category_id'] = name_to_id_map[event['category']]
        del event['category']

    return event_list

def generate_hmac_sha256_signature(secret_key, message):
    # Convert both secret and message to bytes
    key_bytes = secret_key.encode('utf-8')
    message_bytes = message.encode('utf-8')

    # Create HMAC SHA256 signature
    signature = hmac.new(key_bytes, message_bytes, hashlib.sha256).hexdigest()
    return signature

def invoke_webhook(event_list):
    webhook_url = "https://statement-classifier.vercel.app/transactions/webhook"
    raw_json_string = json.dumps(event_list, separators=(',', ':'))
    signature = generate_hmac_sha256_signature(os.getenv("WEBHOOK_SIGNATURE_KEY"), raw_json_string)
    headers = {"Content-Type": "application/json", "x-signature": signature}

    response = requests.post(webhook_url, json={"events": event_list}, headers=headers)

    if response.status_code == 200:
        print("Webhook successfully invoked.")
    else:
        print(f"Webhook invocation failed: {response.status_code} - {response.text}")


app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins; replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/hello")
def say_hello():
    return {"message": "Hello from the named API!"}

class ClassifierRequest(BaseModel):
    client_id: str
    signed_url: str
    file_id: str
    accountant_id: str
    
@app.post("/classifier")
async def classifier_api(request: ClassifierRequest):
    file_list = []
    # download files using presigned urls
    # for file_id, url in file_dict.items():
    file_list.append(extract_text_from_url(request.signed_url))
    # fetch client info from supabase db
    client_info = fetch_supabase_db(request.client_id)
    # call classifier_main - async call
    asyncio.create_task(classifier_main(file_list, client_info['first_name'], client_info['phone_number'], request.client_id, request.file_id, request.accountant_id))
    # return true or false
    return True
    
async def classifier_main(file_list, name, mob_no, client_id, file_id, accountant_id):
    res_final = pd.DataFrame() 
    ## deepseek
    client = OpenAI(
      base_url= "https://openrouter.ai/api/v1",
      api_key= os.getenv("OPENAI_API_KEY"), # Deepseek free chat
    )
    model = "deepseek/deepseek-chat-v3.1:free"
    for file in file_list:
        df = pdf_to_csv(file, client, model)
        res = categorize_transactions_batch(client, df, amount_threshold=150, batch_size=50, model = model, person_name=name, mobile_numbers = mob_no)

        res_final = pd.concat([res_final, res], ignore_index=True)
            
    
    # convert response to webhook event type
    # invoke webhook event
    event_list = df_to_event_list(res_final, client_id, file_id, accountant_id)
    invoke_webhook(event_list)

    return res_final

