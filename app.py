import os # Biblioteca para acessar variáveis de ambiente
import streamlit as st # Framework para criar a interface web do chatbot
from dotenv import load_dotenv # Função que carrega as variáveis do arquivo .env
from langchain_community.document_loaders import PyPDFLoader # Carrega documentos em formato PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter # Divide textos longos em trechos menores (chunks)
from langchain_chroma import Chroma # Banco de dados vetorial utilizado no projeto
from langchain_groq import ChatGroq # Integra o Python à API da Groq para utilizar o LLM
from langchain_huggingface import HuggingFaceEmbeddings # Gera embeddings utilizando modelos do Hugging Face
from pathlib import Path

#1. CONFIGURAÇÃO DA API GROQ

# Carrega as variáveis armazenadas no arquivo .env

load_dotenv()


# Verifica se a chave GROQ_API_KEY está disponível

api_key =  st.secrets.get(
    "GROQ_API_KEY",
    os.getenv("GROQ_API_KEY")
)

if not api_key:

    # Exibe uma mensagem de erro na interface Streamlit
    st.error("Configure a chave GROQ_API_KEY no arquivo .env")

    # Interrompe a execução da aplicação
    st.stop()

#2. INTERFACE INICIAL

# Exibe o título principal da aplicação
st.title("MEI/Receita Federal ")

# Exibe uma mensagem de orientação ao usuário
st.write("Faça perguntas sobre MEI ou receita federal.")

#Para testar: python -m streamlit run app.py
#ctrl c para parar a execução do streamlit

#3. CARREGAMENTO DO DOCUMENTO PDF


#9. PREPARAÇÃO DO RAG

# Armazena em cache os recursos criados pela função
# Evita reconstruir o banco vetorial a cada interação

@st.cache_resource
def iniciar_rag(arquivos_pdf):

    documentos = []

    # Carrega todos os PDFs
    for caminho_pdf, _, _ in arquivos_pdf:

        carregador = PyPDFLoader(caminho_pdf)

        documentos.extend(carregador.load())

    if not documentos:
        raise ValueError("Nenhum PDF carregado.")

    # Divide os documentos em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documentos)

    # Cria os embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Cria o banco vetorial
    banco = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Configura o retriever
    retriever = banco.as_retriever(
        search_kwargs={"k": 5}
    )

    # Configura o LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key
    )

    return retriever, llm


#10. INICIALIZAÇÃO DO CHATBOT

try:

    # Localiza a pasta dos PDFs
    pasta_pdfs = Path(__file__).resolve().parent / "pdfs"

    # Cria a pasta caso ela não exista
    pasta_pdfs.mkdir(exist_ok=True)

    # Lista todos os PDFs encontrados
    arquivos_pdf = sorted(pasta_pdfs.glob("*.pdf"))

    # Verifica se existem arquivos
    if not arquivos_pdf:
        st.error("Nenhum PDF encontrado na pasta pdfs.")
        st.stop()

    # Cria uma assinatura dos arquivos para controlar o cache
    assinatura_pdfs = tuple(
        (
            str(arquivo.resolve()),
            arquivo.stat().st_mtime_ns,
            arquivo.stat().st_size
        )
        for arquivo in arquivos_pdf
    )

    # Prepara o RAG
    with st.spinner("Preparando o chatbot..."):

        retriever, llm = iniciar_rag(assinatura_pdfs)

except Exception as erro:

    st.error(f"Erro ao iniciar: {erro}")
    st.stop()


#11. HISTÓRICO DE MENSAGENS

# Verifica se a lista de mensagens já existe na sessão
if "mensagens" not in st.session_state:

    # Cria uma lista vazia para armazenar o histórico
    st.session_state.mensagens = []


# Percorre as mensagens armazenadas no histórico
for mensagem in st.session_state.mensagens:

    # Cria um balão de conversa para cada mensagem
    # "user" representa o usuário
    # "assistant" representa o chatbot
    with st.chat_message(mensagem["role"]):

        # Exibe o conteúdo da mensagem na interface
        st.markdown(mensagem["content"])


#12. RECEBER A PERGUNTA DO USUÁRIO

# Exibe o campo para o usuário digitar sua pergunta
if pergunta := st.chat_input("Digite sua pergunta..."):

    # Exibe a pergunta na interface do chatbot
    with st.chat_message("user"):
        st.write(pergunta)

    # Armazena a pergunta no histórico de mensagens
    st.session_state.mensagens.append({"role": "user","content": pergunta})

    
    #13. RECUPERAR INFORMAÇÕES E GERAR RESPOSTA

    # Cria o espaço para a resposta do chatbot
    with st.chat_message("assistant"):

        # Exibe uma mensagem durante o processamento
        with st.spinner("Consultando os documentos..."):

            try:
                # Recupera os chunks relevantes para a pergunta
                documentos = retriever.invoke(pergunta)

                # monta o contexto com as referencias
                contexto = ""
                
                for doc in documentos:
                    
                    #recupera o nome do arquivo
                    nome_pdf = Path(doc.metadata["source"]).name
                    
                    #recupera o numero da pagina (começando em zero)
                    pagina = doc.metadata.get("page", 0) + 1
                    
                    contexto +- f"""
                    Documento: {nome_pdf}
                    Página: {pagina}
                    
                    Conteúdo:
                    {doc.page_content}
                    
                    --------------------------------
                    """
                                        
                # Monta as instruções enviadas ao modelo de linguagem
                prompt = f"""
                Você é um assistente especializado em MEI e Receita Federal.

                Responda utilizando apenas as informações
                dos documentos fornecidos no contexto.

                Regras:
                1. Responda de forma clara e objetiva.
                2. Não invente informações.
                3. Se não encontrar a resposta, informe que
                a informação não foi localizada nos documentos.
                4. Sempre informe o nome do documento e a página
                de onde retirou a informação.
                5. Utilize o formato de referência:
                [Fonte: nome_do_arquivo.pdf, página X].


                Contexto:
                {contexto}

                Pergunta:
                {pergunta}
                """

                # Envia o prompt ao LLM e recebe a resposta
                resposta = llm.invoke(prompt).content

                # Exibe a resposta na interface
                st.markdown(resposta)
                
                # mostra as fontes separadamente
                with st.expander("Ver documentos consultados"):
                    
                    fontes = set()
                    
                    for doc in documentos:
                        nome_pdf = Path(doc.Metadata["source"]).name
                        pagina = doc.metadata.get("page", 0)+1
                        
                        fontes.add((nome_pdf, pagina))
                        
                    for nome, pagina in fontes:
                        st.write(f"📄 {nome} — Página {pagina}")

                # Armazena a resposta no histórico
                st.session_state.mensagens.append({"role": "assistant","content": resposta})

            except Exception as erro:
                st.error(f"Erro ao responder: {erro}")
                st.exception(erro)