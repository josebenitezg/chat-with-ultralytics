from dotenv import load_dotenv
from langchain_community.vectorstores import DeepLake
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()

database = DeepLake("hub://intuitivo/ultralytics", embedding=embeddings)
