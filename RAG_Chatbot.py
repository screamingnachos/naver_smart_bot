import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
st.title('Hey I am NAVER Smart Store Bot')
OPENAI_API_KEY = st.text_input('Enter your Open AI API KEY', 'API KEY')
DESTINATION_DIR= st.text_input('Enter your destination file dir', 'DIR')
if st.button("Set API KEY"): 
    st.write("Your OPEN AI API KEY IS" , OPENAI_API_KEY)
    os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY




csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")


if csv_file_uploaded is not None:
    def save_file_to_folder(uploadedFile):
        save_folder = DESTINATION_DIR
        save_path = Path(save_folder, uploadedFile.name)
        with open(save_path, mode='wb') as w:
            w.write(uploadedFile.getvalue())

        if save_path.exists():
            st.success(f'File {uploadedFile.name} is successfully saved!')
            
    save_file_to_folder(csv_file_uploaded)
    loader = CSVLoader(file_path=os.path.join(DESTINATION_DIR, csv_file_uploaded.name))

    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])

    chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")


    st.title("Chat with NAVER Smart Store Bot here")

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []


    def generate_response(user_query):
        response = chain({"question": user_query})
        return response['result']
    
    
    def get_text():
        input_text = st.text_input("You: ","Hi NAVER Smart Store Bot. I have fed in the data. Can you help me with some questions regarding NAVER Smart Store", key="input")
        return input_text
    user_input = get_text()

    if user_input:
        output = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


