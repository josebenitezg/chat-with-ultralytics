from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from database import database

load_dotenv()


def load_markdown_documents(path: str) -> list[Document]:
    loader = DirectoryLoader(path, "**/*.md")
    documents = loader.load()
    return documents


def split_markdown_headings(documents: list[Document]) -> list[Document]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=True)
    chunks: list[Document] = []
    for document in documents:
        chunks.extend(text_splitter.split_text(document.page_content))
    return chunks


def split_paragraphs(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(documents)
    return documents


documents = load_markdown_documents("/Users/olrtg/i/ultralytics/docs/en/")
markdown_chunks = split_markdown_headings(documents)
paragraph_chunks = split_paragraphs(markdown_chunks)

ids = database.add_documents(paragraph_chunks)

for id in ids:
    print(id)

print("Done!")
