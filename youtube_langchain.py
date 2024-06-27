from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    '''
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 100 and 
    k to 4 maximizes the number of tokens to analyze
    '''
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.2)

    #Template to use for the system message prompt
    template = """
        You are a helpful assistant that can answer questions about youtube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question,
        say "I don't know".
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
    db = create_db_from_youtube_video_url(video_url)

    query = "Who are the hosts of this podcast?"
    response, docs = get_response_from_query(db, query)
    print(textwrap.fill(response, width=50))

    
      