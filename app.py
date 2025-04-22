import streamlit as st
import re
import os
import json
import bcrypt
import random
import string
from datetime import datetime
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Buat folder chats dan data kalau belum ada
if not os.path.exists("chats"):
    os.makedirs("chats")
if not os.path.exists("data"):
    os.makedirs("data")

# Inisialisasi users.json kalau belum ada
users_file = "data/users.json"
if not os.path.exists(users_file):
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

# Fungsi untuk load users
def load_users():
    with open(users_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Fungsi untuk save users
def save_users(users):
    with open(users_file, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

# Fungsi untuk register user
def register_user(username, password, role):
    if not username or not password:
        return False, "Username dan password gak boleh kosong!"
    users = load_users()
    if username in users:
        return False, "Username sudah ada!"
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users[username] = {"password": hashed.decode("utf-8"), "role": role}
    save_users(users)
    return True, "Registrasi berhasil! Silakan login."

# Fungsi untuk login user
def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "Username tidak ditemukan!"
    if bcrypt.checkpw(password.encode("utf-8"), users[username]["password"].encode("utf-8")):
        return True, "Login berhasil!"
    return False, "Password salah!"

# Fungsi untuk reset password
def generate_verification_code():
    return ''.join(random.choices(string.digits, k=6))

# Inisialisasi session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
if "user" not in st.session_state:
    st.session_state.user = None
if "reset_code" not in st.session_state:
    st.session_state.reset_code = None
if "reset_username" not in st.session_state:
    st.session_state.reset_username = None

# CSS untuk UI
st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
    }
    .user {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 10px;
        display: flex;
        align-items: center;
    }
    .assistant {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 10px;
        display: flex;
        align-items: center;
    }
    .avatar {
        font-size: 20px;
        margin-right: 10px;
    }
    .chat-text {
        flex: 1;
    }
    .save-button {
        background-color: #007bff;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        cursor: pointer;
    }
    .save-button:hover {
        background-color: #0056b3;
    }
    .reset-button {
        background-color: #6c757d;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        cursor: pointer;
    }
    .reset-button:hover {
        background-color: #5a6268;
    }
    .logout-button {
        background-color: #dc3545;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        cursor: pointer;
    }
    .logout-button:hover {
        background-color: #c82333;
    }
    </style>
""", unsafe_allow_html=True)

# UI login/register/lupa password
if not st.session_state.user:
    st.title("SMARING AI - Login/Register")
    tab1, tab2, tab3 = st.tabs(["Login", "Register", "Lupa Password"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            success, message = login_user(login_username, login_password)
            if success:
                st.session_state.user = {"username": login_username, "role": load_users()[login_username]["role"]}
                st.session_state.reset_code = None
                st.session_state.reset_username = None
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_role = st.selectbox("Role", ["siswa", "guru"], key="reg_role")
        if st.button("Register"):
            success, message = register_user(reg_username, reg_password, reg_role)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with tab3:
        st.subheader("Lupa Password")
        reset_username = st.text_input("Username", key="reset_username")
        if st.session_state.reset_code is None:
            if st.button("Kirim Kode Verifikasi"):
                users = load_users()
                if reset_username in users:
                    st.session_state.reset_code = generate_verification_code()
                    st.session_state.reset_username = reset_username
                    st.success(f"Kode verifikasi: {st.session_state.reset_code} (catat, ini cuma muncul sekali!)")
                else:
                    st.error("Username tidak ditemukan!")
        else:
            st.write(f"Masukkan kode verifikasi untuk {st.session_state.reset_username}")
            verification_code = st.text_input("Kode Verifikasi", key="verification_code")
            new_password = st.text_input("Password Baru", type="password", key="new_password")
            if st.button("Reset Password"):
                if verification_code == st.session_state.reset_code:
                    if new_password:
                        users = load_users()
                        hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
                        users[st.session_state.reset_username]["password"] = hashed.decode("utf-8")
                        save_users(users)
                        st.session_state.reset_code = None
                        st.session_state.reset_username = None
                        st.success("Password berhasil direset! Silakan login.")
                        st.rerun()
                    else:
                        st.error("Password baru gak boleh kosong!")
                else:
                    st.error("Kode verifikasi salah!")
else:
    # Load dokumen
    try:
        loader = TextLoader("data/jadwal.txt")
        documents = loader.load()
    except:
        st.error("File jadwal.txt gak ada! Taruh file di folder data/")
        st.stop()

    # Split dokumen
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Buat vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # Inisialisasi TinyLlama
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        device=-1
    )

    # Buat chain Q&A dengan memori
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        chain_type="stuff"
    )

    # Fungsi untuk update jadwal atau silabus
    def update_file(command, content, file_path):
        if command.lower() == "tambah":
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")
            return f"Berhasil tambah: {content}"
        elif command.lower() == "update":
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Berhasil update: {content}"
        return "Perintah gak dikenal. Pakai 'tambah' atau 'update'."

    # Fungsi untuk edit profil
    def edit_profile(username, new_password, new_role):
        if not new_password and not new_role:
            return False, "Isi setidaknya password atau role baru!"
        users = load_users()
        if new_password:
            hashed = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt())
            users[username]["password"] = hashed.decode("utf-8")
        if new_role:
            users[username]["role"] = new_role
        save_users(users)
        return True, "Profil berhasil diupdate!"

    # Streamlit UI Chat
    st.title("SMARING AI - Asisten Sekolah 🧠")
    st.write(f"Selamat datang, {st.session_state.user['username']} ({st.session_state.user['role']})! Tanya soal jadwal, pelajaran, atau tambah/update data sekolah! 😎")

    # Tampilin riwayat chat
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="user"><span class="avatar">🧑</span><div class="chat-text">**Kamu**: {chat["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="assistant"><span class="avatar">🤖</span><div class="chat-text">**SMARING AI**: {chat["content"]}</div></div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

    # Input user
    user_input = st.text_input("Kamu: ", key="input")
    if user_input:
        with st.spinner("SMARING AI lagi mikir..."):
            # Simpan input user
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Cek perintah tambah/update jadwal atau silabus (hanya untuk guru)
            response = ""
            if st.session_state.user["role"] == "guru":
                jadwal_match = re.match(r"(tambah|update)\s+jadwal\s+(.+)", user_input, re.IGNORECASE)
                silabus_match = re.match(r"(tambah|update)\s+silabus\s+(.+)", user_input, re.IGNORECASE)
                if jadwal_match:
                    command, content = jadwal_match.groups()
                    response = update_file(command, content, "data/jadwal.txt")
                elif silabus_match:
                    command, content = silabus_match.groups()
                    response = update_file(command, content, "data/silabus.txt")
            if not response:
                # Jalankan Q&A normal
                response = qa_chain({"question": user_input})["answer"]

            # Simpan jawaban
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Refresh UI
            st.rerun()

    # Tombol simpan dan reset dalam columns
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.chat_history:
            chat_content = "\n".join(
                f"[{'Kamu' if chat['role'] == 'user' else 'SMARING AI'}] {chat['content']}"
                for chat in st.session_state.chat_history
            )
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"chat_{timestamp}.txt"
            st.download_button(
                label="Simpan Chat",
                data=chat_content,
                file_name=filename,
                mime="text/plain",
                key="save_chat",
                help="Download riwayat chat sebagai file .txt",
                use_container_width=True,
                type="primary"
            )
    with col2:
        st.button(
            "Reset Obrolan",
            key="reset_chat",
            help="Hapus riwayat chat dan mulai dari awal",
            use_container_width=True
        )
        if st.session_state.get("reset_chat"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()

    # Sidebar: User info, edit profil, logout
    with st.sidebar:
        st.write(f"**User**: {st.session_state.user['username']}")
        st.write(f"**Role**: {st.session_state.user['role']}")
        if st.session_state.user["role"] == "guru":
            st.write("Kamu bisa tambah/update jadwal atau silabus!")
        else:
            st.write("Tanya apa aja, simpan chatmu!")
        
        st.markdown("---")
        st.subheader("Edit Profil")
        edit_password = st.text_input("Password Baru (kosongkan jika tidak ubah)", type="password", key="edit_password")
        edit_role = st.selectbox("Role Baru", ["siswa", "guru"], index=["siswa", "guru"].index(st.session_state.user["role"]), key="edit_role")
        if st.button("Update Profil"):
            success, message = edit_profile(st.session_state.user["username"], edit_password, edit_role)
            if success:
                st.session_state.user["role"] = edit_role
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        st.markdown("---")
        if st.button("Logout", key="logout"):
            st.session_state.user = None
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.session_state.reset_code = None
            st.session_state.reset_username = None
            st.rerun()

        st.markdown("---")
        st.write("Tips:")
        st.write("- Tanya: 'Jadwal Senin apa?'")
        if st.session_state.user["role"] == "guru":
            st.write("- Tambah jadwal: 'Tambah jadwal Senin: Fisika 10:00-11:30'")
            st.write("- Update silabus: 'Tambah silabus Matematika: Bab 3 Trigonometri'")
        st.write("- Simpan obrolan pake tombol Simpan Chat")