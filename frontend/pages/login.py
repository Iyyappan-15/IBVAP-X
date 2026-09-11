import streamlit as st
from backend.auth.auth import verify_password, AuthManager
from backend.db.session import SessionLocal
from backend.db.repository import Repository
from backend.interfaces import UserRole

st.set_page_config(page_title="Login — IBVAP-X", page_icon="🔑", layout="centered")

st.title("🔑 IBVAP-X Operator Login")
st.caption("Role-Based Security Operations Governance Portal")

st.markdown("---")

if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "current_role" not in st.session_state:
    st.session_state["current_role"] = None

if st.session_state["auth_token"]:
    st.success(f"Logged in as: **{st.session_state['current_user']}** (Role: `{st.session_state['current_role']}`)")
    if st.button("🚪 Logout"):
        st.session_state["auth_token"] = None
        st.session_state["current_user"] = None
        st.session_state["current_role"] = None
        st.rerun()
else:
    with st.form("login_form"):
        username = st.text_input("Username", value="bop_operator")
        password = st.text_input("Password", type="password", value="demo1234")
        submitted = st.form_submit_button("🔑 Login")

        if submitted:
            db = SessionLocal()
            repo = Repository(db)
            user = repo.get_user_by_username(username)
            db.close()

            if user and verify_password(password, user.hashed_password):
                auth_mgr = AuthManager()
                token = auth_mgr.create_session(username, user.role)
                
                st.session_state["auth_token"] = token
                st.session_state["current_user"] = user.username
                st.session_state["current_role"] = user.role.value
                
                st.success(f"Login successful! Welcome, {user.username}.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

st.markdown("---")
st.info(
    "💡 **Demo Accounts:**\n"
    "- `bop_operator` / `demo1234` (BOP Patrol Operator)\n"
    "- `cmd_operator` / `demo5678` (Command Centre Supervisor)\n"
    "- `admin` / `admin9012` (System Administrator)"
)
