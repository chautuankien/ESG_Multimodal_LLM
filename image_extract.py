from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image

import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
import math

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os
import base64

# Load OpenAI API key from .env file
load_dotenv()
# Fetch the API key
api_key = os.getenv("OPENAI_API_KEY")

report_path = "./data/files/Global_ESG_Q1_2024_Flows_Report.pdf"

report_raw_data = partition_pdf(
    filename=report_path,
    strategy="hi_res",  # mandatory to infer tables
    extract_images_in_pdf=True,
    extract_image_block_to_payload=False,   # if true, will extract base64 for API usage
    extract_image_block_output_dir="./data/files/images"
)
# print(report_raw_data)

"""
### Extract Image Components
def extract_image_metadata(report_data, src_doc):
    image_data = []

    for element in report_data:
        if isinstance(element, Image):
            page_number = element.metadata.page_number
            image_path = element.metadata.image_path if hasattr(element.metadata, "image_path") else None

            image_data.append({
                "source_document": src_doc,
                "page_number": page_number,
                "image_path": image_path
            })
    
    return image_data

extracted_image_data = extract_image_metadata(report_raw_data, report_path)
print(extracted_image_data)



def display_images_from_metadata(extracted_image_data, images_per_row=4):
    valid_images = [img for img in extracted_image_data if img["image_path"]]
    if not valid_images:
        print("No valid image data available")
        return

    num_images = len(valid_images)
    num_rows = math.ceil(num_images / images_per_row)   # Round a number upward to its nearest integer

    fig, axes = plt.subplots(num_rows, num_images, figsize=(20, 5*num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes]

    for ax, image_data in zip(axes, valid_images):
        try:
            img = PIL_Image.open(image_data["image_path"])
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Page {image_data['page_number']}", fontsize=10)
        except Exception as e:
            print(f"Error loading image {image_data['image_path']}: {str(e)}")
            ax.text(0.5, 0.5, f"Error loading image\n{str(e)}", ha='center', va='center')
            ax.axis('off')
    plt.show()
        
display_images_from_metadata(extracted_image_data)
"""

### Image Summarization
images_summarizer_prompt = """
As an ESG analyst for emerging markets investments, please provide a clear interpretation of data or information that see describe from the image.
Focus on ESG-relevant content (Environmental, Social, Governance) and any emerging market context. Describe the type of visual (e.g., chart, photograph, infographic) and its key elements.
Highlight significant data points or trends that are relevant to investment analysis. Avoid bullet points; instead, deliver a coherent, factual summary that captures the essence of the image for ESG investment decision-making.

Image: {image_element}

Limit your description to 3-4 sentences, ensuring it's precise and informative for ESG analysis."""

description_model = ChatOpenAI(api_key=api_key)

def extract_image_metadata_with_summary(report_data, src_doc, images_summarizer_prompt, description_model):
    image_data = []
    prompt = ChatPromptTemplate.from_template(images_summarizer_prompt)

    for element in report_data:
        if isinstance(element, Image):
            page_number = element.metadata.page_number
            image_path = element.metadata.image_path if hasattr(element.metadata, "image_path") else None

            if image_path:
                messages = prompt.format_messages(image_element=image_path)
                image_summary = description_model.invoke(messages).content

                # Read image file and encode it to base64
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                image_data.append({
                    "source_document": src_doc,
                    "page_number": page_number,
                    "image_path": image_path,
                    "image_summary": image_summary,
                    "base64_encoded_image": encoded_image
                })
            else:
                print(f"Warning: Image file not found or path not available for page {page_number}")

    return image_data

extracted_image_data_with_summary = extract_image_metadata_with_summary(report_raw_data, report_path, images_summarizer_prompt, description_model)
print(extracted_image_data_with_summary[0])