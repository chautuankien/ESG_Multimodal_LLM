from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()
# Fetch the API key
api_key = os.getenv("OPENAI_API_KEY")


report_path = "./data/files/Global_ESG_Q1_2024_Flows_Report.pdf"

# Install propper, then add to PATH, link: https://github.com/oschwartz10612/poppler-windows?tab=readme-ov-file
# Install tesseract-ocr then add to PATH, link: https://github.com/UB-Mannheim/tesseract?tab=readme-ov-file

report_raw_data = partition_pdf(
    filename=report_path,
    strategy="hi_res",  # mandatory to infer tables
    extract_images_in_pdf=True,
    extract_image_block_to_payload=False,   # if true, will extract base64 for API usage
    extract_image_block_output_dir="./data/files/images"
)
# print(report_raw_data)
"""
### Extract Table Components
def extract_table_metadata(report_data, src_doc):
    table_data = []

    for element in report_data:
        if isinstance(element, Table):
            page_number = element.metadata.page_number
            table_data.append({
                "source_document": src_doc,
                "page_number": page_number,
                "table_content": str(element)
            })
    
    return table_data

extracted_table_data = extract_table_metadata(report_raw_data, report_path)
print(extracted_table_data)
"""
### Table Summarization
tables_summarizer_prompt = """
As an ESG analyst for emerging markets investments, provide a concise and exact summary of the table contents.
Focus on key ESG metrics (Environmental, Social, Governance) and their relevance to emerging markets.
Highlight significant trends, comparisons, or outliers in the data. Identify any potential impacts on investment strategies or risk assessments.
Avoid bullet points; instead, deliver a coherent, factual summary that captures the essence of the table for ESG investment decision-making.

Table: {table_content}

Limit your summary to 3-4 sentences, ensuring it's precise and informative for ESG analysis in emerging markets."""

description_model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")

def extract_table_metadata_with_summary(report_data, src_doc, tables_summarizer_prompt, description_model):
    table_data = []
    prompt = ChatPromptTemplate.from_template(tables_summarizer_prompt)

    for element in report_data:
        if isinstance(element, Table):
            page_number = element.metadata.page_number
            table_content = str(element)

            messages = prompt.format_messages(table_content=table_content)
            table_summary = description_model.invoke(messages).content
            table_data.append({
                "source_document": src_doc,
                "page_number": page_number,
                "table_content": table_content,
                "table_summary": table_summary
            })

    return table_data

extracted_table_data_with_summary = extract_table_metadata_with_summary(report_raw_data, report_path, 
                                                                        tables_summarizer_prompt, description_model)
print(extracted_table_data_with_summary[0])

