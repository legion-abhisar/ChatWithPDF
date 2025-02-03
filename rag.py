from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.vectorstores import FAISS


class ChatPDF:
    vector_store = None  # Placeholder for the vector store
    retriever = None  # Placeholder for the retriever
    chain = None  # Placeholder for the chain

    def __init__(self):
        # Initialize the chat model with the "mistral" model
        self.model = ChatOllama(model="mistral")
        # Initialize the text splitter with chunk size 1024 and overlap 100
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # Define the prompt template for the assistant
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        # Load the PDF document
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(docs)
        # Filter out unsupported metadata types
        chunks = filter_complex_metadata(chunks)

        # Create a vector store from the document chunks using FastEmbedEmbeddings
        vector_store = FAISS.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # Create a retriever from the vector store with similarity score threshold search
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 1,  # Retrieve top 1 documents
                "score_threshold": 0.2,  # Lowered threshold for better recall
            },
        )

        # Create a chain that processes the context and question through the model
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        # Check if the chain is initialized
        if not self.chain:
            return "Please, add a PDF document first."

        # Invoke the chain with the query and return the result
        return self.chain.invoke(query)

    def clear(self):
        # Clear the vector store, retriever, and chain
        self.vector_store = None
        self.retriever = None
        self.chain = None