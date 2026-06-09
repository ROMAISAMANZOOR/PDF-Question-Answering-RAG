from langchain_community.document_loaders.pdf import PyPDFLoader

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


loaded_documents = load_pdf("C:/Users/jinnah computer/Documents/simple rag project/data/sample.pdf")
print(loaded_documents)