import weaviate
import weaviate.classes.config as wc
import os

from dotenv import load_dotenv

load_dotenv()
wcs_url = os.getenv("WEAVIATE_URL")
wcs_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a Weaviate client
client = weaviate.connect_to_weaviate_cloud(cluster_url=wcs_url, auth_credentials=weaviate.auth.AuthApiKey(wcs_api_key),
                              headers={
                                  "X-OpenAI-Api-Key": openai_api_key
                              })

print(client.is_ready())

#### Create multimodal ESG collection
"""
A collection is created by specifying the following parameters:
    name: name of the collection.
    properties: list of all attributes of the collection.
    vectorizer_config: details of the embedding model to use
"""

# skip_vectorization=True for properties that do not require vectorization.
# Only the properties such as text data, image description, audio transcription, and table description require vectorization for performing a search.
properties = [
    wc.Property(name="source_document", data_type=wc.DataType.TEXT, skip_vectorization=True),
    wc.Property(name="page_number", data_type=wc.DataType.INT, skip_vectorization=True),
    wc.Property(name="paragraph_number", data_type=wc.DataType.INT, skip_vectorization=True),
    wc.Property(name="text", data_type=wc.DataType.TEXT),
    wc.Property(name="image_path", data_type=wc.DataType.TEXT, skip_vectorization=True),
    wc.Property(name="description", data_type=wc.DataType.TEXT),
    wc.Property(name="base64_encoding", data_type=wc.DataType.BLOB, skip_vectorization=True),
    wc.Property(name="table_content", data_type=wc.DataType.TEXT),
    wc.Property(name="url", data_type=wc.DataType.TEXT, skip_vectorization=True),
    wc.Property(name="audio_path", data_type=wc.DataType.TEXT, skip_vectorization=True),
    wc.Property(name="transcription", data_type=wc.DataType.TEXT),
    wc.Property(name="content_type", data_type=wc.DataType.TEXT, skip_vectorization=True),
]

client.collections.create(
    name="ESGDocuments",
    property=properties,
    vectorizer_config=None
)