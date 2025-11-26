import streamlit as st
import os
import json
import tempfile
from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field

# ==========================================
# 1. KONFIGURASI & CSS
# ==========================================
st.set_page_config(page_title="PAKAR - AI Career", page_icon="🎓")

st.markdown("""
<style>
    .stApp {background-color: #0e1117;}
    
    div.block-container {
        padding-top: 2rem;       
        padding-bottom: 2rem;    
        padding-left: 2rem;      
        padding-right: 2rem;     
        max-width: 100%;         
    }

    /* --- HERO SECTION (JUDUL) --- */
    .hero-title {
        font-size: 3.5rem !important; 
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.2rem !important; 
        color: #c9d1d9; 
        margin-bottom: 2rem; 
        max-width: 800px; 
    }
    
    /* Profile Section */
    .profile-container {
        background-color: #161b22; border: 1px solid #30363d;
        padding: 25px; border-radius: 12px; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .profile-name { font-size: 2rem; font-weight: bold; color: #f0f6fc; margin: 0; }
    .profile-degree { color: #8b949e; font-size: 1.1rem; margin-bottom: 15px; }
    .profile-summary { color: #c9d1d9; font-style: italic; border-left: 3px solid #58a6ff; padding-left: 15px; line-height: 1.6;}

    /* Skill Badges */
    .skill-tag {
        display: inline-block;
        background-color: #21262d;
        color: #58a6ff;
        border: 1px solid #30363d;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
        transition: all 0.2s;
    }
    .skill-tag:hover { background-color: #58a6ff; color: white; border-color: #58a6ff; cursor: default; }

    /* Job Cards */
    .job-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: transform 0.2s, border-color 0.2s;
        display: flex; flex-direction: column;
    }
    .job-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
        box-shadow: 0 10px 20px rgba(88, 166, 255, 0.1);
    }
    .job-title { font-size: 1.3rem; font-weight: bold; color: #f0f6fc; margin-bottom: 10px; min-height: 60px; display: flex; align-items: center; }
    .match-badge {
        font-size: 0.85rem; font-weight: bold;
        padding: 4px 10px; border-radius: 6px;
        margin-bottom: 10px; display: inline-block;
    }
    .job-desc { font-size: 0.9rem; color: #8b949e; margin-top: 15px; line-height: 1.5; flex-grow: 1; }

    /* Custom Progress Bar */
    .progress-bg { background: #30363d; height: 8px; border-radius: 4px; width: 100%; margin-top: 5px; }
    .progress-fill { height: 100%; border-radius: 4px; transition: width 1s ease-in-out; }

    /* Gap Analysis Box */
    .gap-box {
        background-color: #252110; 
        border: 1px solid #9e7700;
        border-left: 5px solid #d29922;
        color: #eac54f;
        padding: 20px;
        border-radius: 8px;
        margin-top: 25px;
    }
    
    /* Interview Score Card */
    .score-card {
        color: white; padding: 15px;
        border-radius: 10px; text-align: center; font-weight: bold; font-size: 32px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Button Custom */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #c9d1d9;
    }
    .stButton > button:hover {
        border-color: #8b949e;
        color: white;
        background-color: #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI DATA (PYDANTIC MODELS)
# ==========================================
class ResumeData(BaseModel):
    nama_kandidat: str = Field(description="Nama lengkap kandidat.")
    pendidikan_tertinggi: str = Field(description="Pendidikan tertinggi.")
    skills_utama: list[str] = Field(description="Daftar 5-10 skill teknis.")
    ringkasan_cv: str = Field(description="Ringkasan eksekutif singkat tentang profil kandidat (2-3 kalimat).")

class JobRecommendation(BaseModel):
    judul_pekerjaan: str = Field(description="Posisi pekerjaan.")
    skor_kecocokan: str = Field(description="Persentase 0-100%.")
    alasan: str = Field(description="Alasan spesifik kenapa cocok (maks 2 kalimat).")

class CareerAnalysis(BaseModel):
    rekomendasi: List[JobRecommendation] = Field(description="Daftar 3 rekomendasi pekerjaan.")
    analisis_gap: str = Field(description="Saran pengembangan skill (gap analysis) yang konkret.")

class InterviewFeedback(BaseModel):
    skor: int = Field(description="Skor jawaban user (0-100).")
    feedback_positif: str = Field(description="Apa yang sudah bagus dari jawaban user.")
    feedback_negatif: str = Field(description="Apa yang perlu diperbaiki/kurang.")
    jawaban_saran: str = Field(description="Contoh jawaban ideal/sempurna untuk pertanyaan tersebut.")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def load_and_read_file(uploaded_file):
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        if uploaded_file.name.endswith('.pdf'): loader = PyPDFLoader(tmp.name)
        elif uploaded_file.name.endswith('.txt'): loader = TextLoader(tmp.name)
        else: return None
        docs = loader.load()
        return " ".join([doc.page_content for doc in docs])
    finally:
        if tmp and os.path.exists(tmp.name): os.remove(tmp.name)

def clean_and_parse_json(content, parser_class):
    try: return parser_class.parse(content)
    except:
        content = content.replace("json", "").replace("```", "").strip()
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            try: return parser_class.pydantic_object(**json.loads(content[start:end+1]))
            except: return None
        return None

# ==========================================
# 4. AI TOOLS & AGENT TOOLS (CORE LOGIC)
# ==========================================

# Helper: Resume Parser
def parse_resume_with_llm(text, api_key):
    parser = PydanticOutputParser(pydantic_object=ResumeData)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Extract resume data to JSON:\n{format_instructions}\nResume:\n{resume_text}"
    ).partial(format_instructions=parser.get_format_instructions())
    res = (prompt | llm).invoke({"resume_text": text})
    return clean_and_parse_json(res.content, parser)

# Helper: Career Analyzer
def analyze_career_path(data, api_key):
    parser = PydanticOutputParser(pydantic_object=CareerAnalysis)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.2)
    profile = f"Nama: {data['nama_kandidat']}, Skill: {data['skills_utama']}, Info: {data['ringkasan_cv']}"
    prompt = ChatPromptTemplate.from_template(
        "Based on this profile, suggest 3 specific job titles and provide a gap analysis in JSON format:\n{profile}\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    res = (prompt | llm).invoke({"profile": profile})
    return clean_and_parse_json(res.content, parser)

# DEFINISI 3 TOOLS AGENT (AGENTIC ARCHITECTURE) 

@tool
def tool_study_plan(skill_name: str):
    """
    Gunakan alat ini jika user meminta rencana belajar, kurikulum, atau roadmap.
    Input: Nama skill. Output: Rencana belajar 4 minggu.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: return "Error: API Key hilang."
        llm_tool = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.7)
        return llm_tool.invoke(f"Buatkan study plan 4 minggu yang ringkas dan padat untuk mempelajari: {skill_name}.").content
    except Exception as e:
        return f"Gagal membuat study plan: {str(e)}"

@tool
def tool_cover_letter(job_title: str):
    """
    Gunakan alat ini jika user meminta bantuan membuat surat lamaran kerja (Cover Letter).
    Input: Judul pekerjaan. Output: Draft surat lamaran kerja.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: return "Error: API Key hilang."
        llm_tool = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.7)
        prompt = (
            f"Buatkan draft Cover Letter (Surat Lamaran Kerja) profesional dalam Bahasa Indonesia "
            f"untuk posisi: {job_title}. "
            f"Buatlah surat yang persuasif, menonjolkan semangat belajar, dan siap berkontribusi. "
            f"Gunakan format [Nama Kandidat], [Perusahaan Tujuan] sebagai placeholder."
        )
        return llm_tool.invoke(prompt).content
    except Exception as e:
        return f"Gagal membuat surat lamaran: {str(e)}"

@tool
def tool_linkedin_optimization(role_target: str):
    """
    Gunakan alat ini jika user meminta saran optimasi profil LinkedIn atau personal branding.
    Input: Target pekerjaan/Role. Output: Saran Headline, About Section, dan Hashtag.
    """
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key: return "Error: API Key hilang."
        llm_tool = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.7)
        prompt = (
            f"Buatkan optimasi profil LinkedIn untuk seseorang yang menargetkan posisi: {role_target}. "
            f"Berikan output berupa:\n"
            f"1. 3 Opsi 'Headline' yang profesional dengan kata kunci SEO.\n"
            f"2. Draft 'About' (Ringkasan Diri) yang engaging (maks 2 paragraf).\n"
            f"3. 5 Hashtag relevan."
        )
        return llm_tool.invoke(prompt).content
    except Exception as e:
        return f"Gagal optimasi LinkedIn: {str(e)}"

# AGENT ORCHESTRATOR 
def get_agent_response(query, history, profile_context, api_key):
    # 1. Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.5)
    
    # 2. DAFTAR 3 TOOLS
    tools = [tool_study_plan, tool_cover_letter, tool_linkedin_optimization] 
    
    # 3. System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Anda adalah AI Career Coach agent profesional. Tugas utama Anda adalah memberikan saran karir strategis. "
         "Anda memiliki akses ke 3 tools khusus: 'Study Plan Generator', 'Cover Letter Drafter', dan 'LinkedIn Optimizer'. "
         "Gunakan tools tersebut JIKA DAN HANYA JIKA user memintanya atau jika konteks percakapan membutuhkannya. "
         "\n\n"
         "ATURAN RESPON PENTING:\n"
         "1. Jika pertanyaan terkait karir/skill: Jawablah dengan profesional dan gunakan tools jika perlu.\n"
         "2. Jika pertanyaan DI LUAR KONTEKS (misal: 'kenapa bumi bulat', 'resep masakan', dll): \n"
         "   - Jawablah pertanyaan tersebut secara ringkas dan sopan.\n"
         "   - NAMUN, di akhir jawaban, Anda WAJIB menambahkan kalimat transisi untuk mengajak user kembali fokus ke topik karir. \n"
         "   (Contoh: '...Tapi ngomong-ngomong, bagaimana progres persiapan karir Anda? Apakah ada yang ingin didiskusikan terkait CV atau skill?')\n"
         "3. Jika user meminta simulasi interview: Arahkan mereka untuk membuka Tab 3 (Simulasi Interview).\n"
         "\n"
         "Informasi Profil User saat ini:\n{context_data}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 4. Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 5. Run Agent
    response = agent_executor.invoke({
        "input": query, 
        "chat_history": history,
        "context_data": profile_context
    })
    return response["output"]

# MANUAL INTERVIEW FUNCTIONS (TAB 3) 
def generate_interview_question(job_title, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.8)
    prompt = f"Buatlah 1 pertanyaan interview yang SULIT, SPESIFIK dan TEKNIS untuk posisi '{job_title}'. Langsung tulis pertanyaannya saja tanpa kalimat pembuka."
    return llm.invoke(prompt).content

def evaluate_interview_answer(question, answer, job_title, api_key):
    parser = PydanticOutputParser(pydantic_object=InterviewFeedback)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        "Anda adalah Senior Interviewer untuk posisi {job_title}.\n"
        "Pertanyaan: {question}\n"
        "Jawaban Kandidat: {answer}\n\n"
        "Berikan penilaian jujur dan detail dalam format JSON:\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    res = (prompt | llm).invoke({"question": question, "answer": answer, "job_title": job_title})
    return clean_and_parse_json(res.content, parser)

# ==========================================
# 5. MAIN APP
# ==========================================
def main():
    # Session State Init
    if "parsed_data" not in st.session_state: st.session_state.parsed_data = None
    if "career_advice" not in st.session_state: st.session_state.career_advice = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "interview_q" not in st.session_state: st.session_state.interview_q = None
    if "interview_job" not in st.session_state: st.session_state.interview_job = None
    if "interview_feedback" not in st.session_state: st.session_state.interview_feedback = None

    # SIDEBAR 
    with st.sidebar:
        st.header("🔑 Akses Sistem")
        api_key = st.text_input("Masukkan Gemini API Key", type="password", help="Dapatkan di aistudio.google.com")
        st.divider()

    # HERO SECTION
    st.markdown('<p class="hero-title">PAKAR</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Platform Analisis Karir berbasis Agent yang menyediakan pemetaan keahlian, pembuatan dokumen karir, dan simulasi wawancara.</p>', unsafe_allow_html=True)

    # Feature Grid
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1: st.info("📄 *Analisis CV*")
    with c2: st.success("🧠 *Pencocokan Karir*")
    with c3: st.warning("🤖 *Fitur Agen AI*")
    with c4: st.error("🎤 *Simulasi Interview*")

    st.divider()

    if not api_key:
        st.warning("⬅ Masukkan API Key di Sidebar untuk memulai.")
        st.stop()
        
    os.environ["GEMINI_API_KEY"] = api_key

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Analisis CV & Karir", "💬 Konsultasi & Documents", "📝 Simulasi Interview"])

    # TAB 1: ANALYZER 
    with tab1:
        st.markdown("### 📂 Upload CV Kandidat")
        uploaded_file = st.file_uploader("Format PDF atau TXT", type=['pdf', 'txt'])
        
        if uploaded_file:
            if st.button("🚀 Mulai Analisis Profil", type="primary", use_container_width=True):
                with st.spinner("🔍 Sedang mengekstrak informasi dan mencocokkan karir..."):
                    text = load_and_read_file(uploaded_file)
                    if text:
                        p_data = parse_resume_with_llm(text, api_key)
                        if p_data:
                            st.session_state.parsed_data = p_data.dict()
                            c_advice = analyze_career_path(p_data.dict(), api_key)
                            if c_advice:
                                st.session_state.career_advice = c_advice.dict()
                                st.session_state.interview_q = None
                                st.session_state.interview_feedback = None
                            else: st.error("Gagal analisis karir.")
                        else: st.error("Gagal parsing CV.")
        
        # Tampilan Hasil Analisis
        if st.session_state.parsed_data and st.session_state.career_advice:
            data = st.session_state.parsed_data
            advice = st.session_state.career_advice
            
            st.markdown("---")
            
            # Profile
            st.markdown(f"""
            <div class="profile-container">
                <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 300px;">
                        <p class="profile-name">{data['nama_kandidat']}</p>
                        <p class="profile-degree">🎓 {data['pendidikan_tertinggi']}</p>
                    </div>
                    <div style="flex: 1.5; min-width: 300px;">
                         <p class="profile-summary">"{data['ringkasan_cv']}"</p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 8px;">🛠 <b>TOP SKILLS DETECTED:</b></p>
                    {' '.join([f'<span class="skill-tag">{s}</span>' for s in data['skills_utama']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Job Recommendations
            st.markdown("### 🎯 Rekomendasi Karir & Kecocokan")
            cols = st.columns(3)
            for i, job in enumerate(advice['rekomendasi']):
                try: score_int = int(job['skor_kecocokan'].replace('%', ''))
                except: score_int = 0
                bar_color = "#238636" if score_int >= 80 else "#d29922" if score_int >= 60 else "#da3633"
                
                with cols[i]:
                    st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{job['judul_pekerjaan']}</div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="match-badge" style="background-color: {bar_color}33; color: {bar_color}; border: 1px solid {bar_color};">
                                Match: {job['skor_kecocokan']}
                            </span>
                        </div>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width: {score_int}%; background-color: {bar_color};"></div>
                        </div>
                        <p class="job-desc">{job['alasan']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Gap Analysis
            st.markdown(f"""
            <div class="gap-box">
                <h4 style="margin:0; margin-bottom: 10px;">🚀 Strategi Pengembangan Karir</h4>
                <p style="margin:0; line-height: 1.6;">{advice['analisis_gap']}</p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: AGENT CHATBOT (DENGAN BUTTON SHORTCUTS)
    with tab2:
        if not st.session_state.parsed_data:
            st.info("⚠ Harap lakukan analisis CV di Tab 1 terlebih dahulu agar Agent memiliki konteks.")
        else:
            st.subheader("💬 Konsultasi & Pembuatan Dokumen")
            st.caption("Agent cerdas yang dapat membuatkan dokumen atau rencana belajar untuk Anda.")

            # QUICK ACTION BUTTONS
            st.markdown("##### ⚡ Tindakan Cepat:")
            b_col1, b_col2, b_col3 = st.columns(3)
            
            clicked_prompt = None
            
            # Mendapatkan data skill dan job untuk prompt otomatis
            first_skill = st.session_state.parsed_data['skills_utama'][0] if st.session_state.parsed_data['skills_utama'] else "Python"
            target_job = st.session_state.career_advice['rekomendasi'][0]['judul_pekerjaan'] if st.session_state.career_advice['rekomendasi'] else "Data Scientist"

            if b_col1.button("📅 Buat Rencana Belajar", use_container_width=True, help=f"Buat study plan untuk {first_skill}"):
                clicked_prompt = f"Buatkan rencana belajar 4 minggu lengkap untuk menguasai skill: {first_skill}. Saya ingin fokus pada praktik."
            
            if b_col2.button("✍️ Draft Cover Letter", use_container_width=True, help=f"Buat surat lamaran untuk {target_job}"):
                clicked_prompt = f"Buatkan draft Cover Letter profesional untuk posisi {target_job}. Tekankan bahwa saya cepat belajar."
                
            if b_col3.button("💼 Optimasi LinkedIn", use_container_width=True, help=f"Saran profil untuk {target_job}"):
                clicked_prompt = f"Berikan saran optimasi profil LinkedIn (Headline & About) agar menarik rekruter untuk posisi {target_job}."

            # History Chat
            for msg in st.session_state.chat_history:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                with st.chat_message(role): st.write(msg.content)
            
            # Input Logic (Gabungan Manual & Button)
            manual_input = st.chat_input("Diskusikan strategi karir Anda di sini...")
            
            # Tentukan input final (prioritas tombol jika diklik, jika tidak maka input manual)
            final_query = clicked_prompt if clicked_prompt else manual_input

            if final_query:
                # 1. Tampilkan pesan user
                with st.chat_message("user"): st.write(final_query)
                st.session_state.chat_history.append(HumanMessage(content=final_query))
                
                # 2. Agent Berpikir & Menjawab
                with st.chat_message("assistant"):
                    with st.spinner("Agent sedang berpikir & memilih tools yang tepat..."):
                        data = st.session_state.parsed_data
                        advice = st.session_state.career_advice
                        context = f"Data Kandidat: {data}\nRekomendasi Sistem: {advice['rekomendasi']}\nGap Analysis: {advice['analisis_gap']}"
                        
                        response = get_agent_response(final_query, st.session_state.chat_history, context, api_key)
                        st.write(response)
                
                # 3. Simpan respon & Rerun untuk update UI
                st.session_state.chat_history.append(AIMessage(content=response))
                st.rerun()

    # TAB 3: MOCK INTERVIEW
    with tab3:
        st.subheader("📝 Ujian Simulasi Interview (Mock Test)")
        st.write("Mode ujian terstruktur dengan penilaian skor otomatis oleh AI.")
        
        if not st.session_state.career_advice:
            st.info("⚠ Harap lakukan analisis CV di Tab 1 terlebih dahulu.")
        else:
            recs = [job['judul_pekerjaan'] for job in st.session_state.career_advice['rekomendasi']]
            selected_job = st.selectbox("Pilih Posisi untuk Ujian:", recs)
            
            if st.button("🎲 Generate Soal Ujian", type="primary"):
                with st.spinner("🤖 AI sedang menyiapkan soal ujian..."):
                    st.session_state.interview_job = selected_job
                    st.session_state.interview_q = generate_interview_question(selected_job, api_key)
                    st.session_state.interview_feedback = None
            
            if st.session_state.interview_q:
                st.markdown("---")
                st.markdown(f"### 🤖 Penguji ({st.session_state.interview_job}) bertanya:")
                st.info(f"🗣 *{st.session_state.interview_q}*")
                
                user_answer = st.text_area("Jawaban Ujian Anda:", height=150, placeholder="Jawablah selengkap mungkin menggunakan metode STAR (Situation, Task, Action, Result)...")
                
                if st.button("📝 Kumpulkan Jawaban"):
                    if not user_answer:
                        st.warning("Harap isi jawaban ujian dulu.")
                    else:
                        with st.spinner("👨‍⚖ AI sedang menilai lembar jawaban..."):
                            feedback = evaluate_interview_answer(
                                st.session_state.interview_q, 
                                user_answer, 
                                st.session_state.interview_job, 
                                api_key
                            )
                            if feedback:
                                st.session_state.interview_feedback = feedback.dict()
                            else:
                                st.error("Gagal melakukan penilaian.")

            if st.session_state.interview_feedback:
                fb = st.session_state.interview_feedback
                st.markdown("---")
                st.markdown("### 📊 Hasil Ujian & Raport")
                
                col_score, col_details = st.columns([1, 3])
                with col_score:
                    st.markdown("Skor Akhir:")
                    color = "#28a745" if fb['skor'] >= 75 else "#ffc107" if fb['skor'] >= 50 else "#dc3545"
                    st.markdown(f"<div class='score-card' style='background-color: {color}'>{fb['skor']}/100</div>", unsafe_allow_html=True)
                
                with col_details:
                    with st.container():
                        st.markdown("✅ *Poin Plus:*")
                        st.write(fb['feedback_positif'])
                        st.markdown("⚠ *Koreksi:*")
                        st.write(fb['feedback_negatif'])
                
                with st.expander("💡 Kunci Jawaban (Saran AI)"):
                    st.info(fb['jawaban_saran'])

if __name__ == "__main__":

    main()
