import streamlit as st
import os
from chatbot_documentos import (
    carregar_credenciais,
    indexar_documento,
    responder_pergunta
)

# =============================
# Configuração do Ambiente
# =============================
try:
    credenciais = carregar_credenciais()
    os.environ["LANGSMITH_API_KEY"] = credenciais["LANGSMITH_API_KEY"]
    os.environ["MISTRAL_API_KEY"] = credenciais["MISTRAL_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"
except ValueError as e:
    st.error(f"Erro ao carregar credenciais: {e}")
    st.stop()

# =============================
# Inicialização do estado da sessão
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# =============================
# Interface com o Usuário
# =============================
st.set_page_config(layout="centered")
st.title("💬 Document Chatbot")

uploaded_file = st.file_uploader("📄 Upload here the PDF Document", type=["pdf"])

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("📚 Indexing document..."):
        caminho_pdf = "document_temp.pdf"
        with open(caminho_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            store = indexar_documento(caminho_pdf)
            st.session_state.vector_store = store
            st.success("✅ Document attached successfully.")
        except Exception as e:
            st.error(f"An error occurred during processing.: {e}")
            st.stop()

# Exibir histórico de chat
#st.markdown("### 💬 Histórico do Chat")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Caixa de entrada estilo chat
if st.session_state.vector_store:
    prompt = st.chat_input("Type your question")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("🤖 Thinking..."):
            try:
                resposta = responder_pergunta(st.session_state.vector_store, prompt)
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                with st.chat_message("assistant"):
                    st.markdown(resposta)
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
