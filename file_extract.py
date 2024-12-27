from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Image, Table

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
import math

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

### Extract Textual Components
def extract_text_metadata(report_data, src_doc):
    text_data = []
    paragraph_counters = {}

    for element in report_data:
        if isinstance(element, NarrativeText):
            page_number = element.metadata.page_number

            if page_number not in paragraph_counters:
                paragraph_counters[page_number] = 1
            else:
                paragraph_counters[page_number] += 1
            
            paragraph_number = paragraph_counters[page_number]
            text_content = element.text

            text_data.append({
                "sorce_document": src_doc,
                "page_number": page_number,
                "paragraph_number": paragraph_number,
                "text": text_content
            })
    
    return text_data

extracted_data = extract_text_metadata(report_raw_data, report_path)
print(extracted_data)





