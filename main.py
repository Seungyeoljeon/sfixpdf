__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#import
from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button

button(username="jocoding", floating=True, width=221)

#제목
streamlit.title("ChatPDF")
streamlit.write("___")

#uploader
uploaded_file = streamlit.file_uploader("Choose a file")
streamlit.write("___")


def pdf_to_document(uploaded_file):
    # 파일 확장자 확인
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # PDF 파일의 경우
    if file_extension == '.pdf':
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath,"wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()
        return pages
    #TXT 파일의 경우
    elif file_extension == '.txt':
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w+") as temp_file:
            temp_file.write(uploaded_file.getvalue().decode("utf-8"))
            temp_file.flush()
            return pdf_to_document(temp_file)

    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are supported.")

def split_txt(txt_file):
    with open(txt_file, "r") as f:
        pages = []
        for line in f:
            pages.append({"page_content": line})
    return pages

# 업로드된 파일이 TXT 파일인 경우
if uploaded_file is not None and file_extension == '.txt':
    pages = split_txt(uploaded_file)
else:
    pages = pdf_to_document(uploaded_file)


#업로드시 동작 코드
if uploaded_file is not None:
    try:
        pages = pdf_to_document(uploaded_file)
        
    except ValueError as e:
        streamlit.error(str(e))
        pages = None
        
if pages is None:
    streamlit.warning("Please upload a valid PDF or TXT file.")
else:
        
    #Split
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)
      
    #Embedding
    
    embeddings_model = OpenAIEmbeddings()
    
    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)
    
    
    # llm = ChatOpenAI(temperature=0)
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    #     retriever=db.as_retriever(), llm=llm
    # )
    
    #input
    streamlit.header("pdf에게 질문하기")
    question = streamlit.text_input('질문입력하세요.')
    
    #question
    if streamlit.button('질문하기'):
        with streamlit.spinner('Wait for it...'):
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            streamlit.write(result["result"])


print(pages)
streamlit.write(pages)
