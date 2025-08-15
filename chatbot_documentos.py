import os
import time
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

# =============================
# Carregar credenciais
# =============================
def carregar_credenciais(dotenv_path: str = ".env"):
    load_dotenv(dotenv_path)
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")

    if not langsmith_key or not mistral_key:
        raise ValueError("Credenciais não encontradas no arquivo .env.")

    return {
        "LANGSMITH_API_KEY": langsmith_key,
        "MISTRAL_API_KEY": mistral_key
    }

# =============================
# Indexar documento PDF (chunking aprimorado)
# =============================
def indexar_documento(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    # Chunking mais coeso para preservar contexto e relações estruturais
    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
    partes = splitter.split_documents(documentos)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    store = InMemoryVectorStore(embeddings)
    store.add_documents(partes)
    return store

# =============================
# Responder com inferência contextual para documentos variados
# =============================
def responder_pergunta(store, pergunta_usuario: str) -> str:
    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

    try:
        trechos = store.similarity_search(pergunta_usuario, k=12)
    except Exception as e:
        raise RuntimeError(f"Erro ao buscar documentos: {e}")

    # Concatena somente os textos limpos dos trechos
    contexto = ""
    for doc in trechos:
        texto = doc.page_content.strip()
        if texto.startswith("["):
            texto = texto.split("\n", 1)[-1]
        contexto += texto.strip() + "\n\n"

    mensagem_sistema = SystemMessage(
        content=(
            "Você é um assistente treinado para analisar e responder perguntas com base no conteúdo de documentos de texto, "
            "como relatórios técnicos, artigos acadêmicos, atas de reuniões, pareceres institucionais, dissertações ou documentos administrativos. "
            "Os trechos abaixo foram extraídos automaticamente e podem não seguir a ordem original. "
            "Seu papel é inferir significados, estruturas, temas e padrões contextuais mesmo que não estejam rotulados. "
            "Evite mencionar que está vendo 'trechos' ou numerá-los. Seja preciso, dedutivo e objetivo. "
            "Não responda subjetivamente a perguntas com base nos techos. Seja sempre objetivo."
            "Se for possível inferir o conteúdo, explique com clareza. Caso contrário, diga que não é possível determinar com segurança.\n\n"
            "Conteúdo extraído:\n" + contexto
        )
    )

    mensagens = [
        mensagem_sistema,
        {"role": "user", "content": pergunta_usuario}
    ]

    for attempt in range(10):
        try:
            resposta = llm.invoke(mensagens)
            return resposta.content.strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"Tentativa {attempt+1}: Limite excedido (429). Aguardando 5 segundos...")
                time.sleep(5)
                continue
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Erro ao processar a pergunta: {e}")
