import os
import yaml
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SCENARIOS_DIR = Path("data/scenarios")
DEFAULT_MODEL = os.getenv("MODEL_DEPLOYMENT_NAME", "gemini-2.5-flash")
BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Pydantic models for structured evaluation output ---

class SpeakingToneStyle(BaseModel):
    professional_tone: int
    active_listening: int
    engagement_quality: int
    total: int

class ConversationContent(BaseModel):
    needs_assessment: int
    value_proposition: int
    objection_handling: int
    total: int

class SalesEvaluation(BaseModel):
    speaking_tone_style: SpeakingToneStyle
    conversation_content: ConversationContent
    overall_score: int
    strengths: List[str]
    improvements: List[str]
    specific_feedback: str

# --- App State Initialization ---

if "messages" not in st.session_state:
    st.session_state.messages = []
if "scenario_id" not in st.session_state:
    st.session_state.scenario_id = None
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "analysis_loading" not in st.session_state:
    st.session_state.analysis_loading = False

# --- Helper Functions ---

def load_scenarios() -> Dict[str, Dict[str, Any]]:
    """Load all role-play scenarios from YAML files."""
    scenarios = {}
    if not SCENARIOS_DIR.exists():
        return scenarios
    
    for file in SCENARIOS_DIR.glob("*-role-play.prompt.yml"):
        try:
            with open(file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                scenario_id = file.stem.replace("-role-play.prompt", "")
                scenarios[scenario_id] = data
        except Exception as e:
            st.error(f"Error loading scenario {file.name}: {e}")
    return scenarios

def load_evaluation_prompt(scenario_id: str) -> str:
    """Load the evaluation prompt for a specific scenario."""
    eval_file = SCENARIOS_DIR / f"{scenario_id}-evaluation.prompt.yml"
    if eval_file.exists():
        with open(eval_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data["messages"][0]["content"]
    return "You are an expert sales coach. Evaluate the conversation."

def get_gemini_client():
    """Initialize the OpenAI client for Gemini."""
    if not API_KEY:
        st.error("GEMINI_API_KEY is not set. Please check your .env file.")
        st.stop()
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

async def analyze_conversation(scenario_id: str, transcript: str):
    """Analyze the conversation using Gemini structured output."""
    client = get_gemini_client()
    eval_prompt_base = load_evaluation_prompt(scenario_id)
    
    prompt = f"""{eval_prompt_base}

    IMPORTANT EVALUATION RULES:
    1. YOUR TARGET: Evaluate the **SALESPERSON** (labeled 'Salesperson' in the transcript).
    2. IGNORE PROSPECT: The 'Prospect' is the AI playing a role. Do NOT evaluate the Prospect's performance, tone, or effectiveness.
    3. PERSPECTIVE: Provide feedback directly to the Salesperson to help them improve.

    EVALUATION CRITERIA:
    - SPEAKING TONE & STYLE (30 points): professional_tone (10), active_listening (10), engagement_quality (10)
    - CONVERSATION CONTENT (70 points): needs_assessment (25), value_proposition (25), objection_handling (20)

    Calculate overall_score as the sum of all individual scores (max 100).
    Provide 3 strengths and 3 improvements for the SALESPERSON.

    CONVERSATION TO EVALUATE:
    {transcript}
    """


    try:
        completion = client.beta.chat.completions.parse(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert sales conversation evaluator."},
                {"role": "user", "content": prompt}
            ],
            response_format=SalesEvaluation,
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

# --- UI Layout ---

st.set_page_config(page_title="GTT Sales Coach", layout="wide", page_icon="🎯")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .score-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .briefing-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4bafff;
        margin-bottom: 20px;
    }
</style>

""", unsafe_allow_html=True)

# --- Sidebar ---

with st.sidebar:
    st.title("🎯 GTT Sales Coach")
    st.markdown("---")
    
    scenarios = load_scenarios()
    scenario_options = {data["name"]: id for id, data in scenarios.items()}
    
    selected_name = st.selectbox("Choose a Scenario", list(scenario_options.keys()))
    new_scenario_id = scenario_options[selected_name]
    
    if st.session_state.scenario_id != new_scenario_id:
        st.session_state.scenario_id = new_scenario_id
        st.session_state.messages = []
        st.session_state.assessment = None
        # Add initial system message from scenario
        system_msg = scenarios[new_scenario_id]["messages"][0]["content"]
        st.session_state.messages.append({"role": "system", "content": system_msg})
        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.assessment = None
        st.rerun()

    if st.button("📊 Analyze Performance", type="primary"):
        if len(st.session_state.messages) <= 1:
            st.warning("Start a conversation first!")
        else:
            st.session_state.analysis_loading = True

    # --- Sidebar Briefing ---
    if st.session_state.scenario_id in scenarios:
        scenario_data = scenarios[st.session_state.scenario_id]
        if "briefing" in scenario_data:
            b = scenario_data["briefing"]
            st.markdown("---")
            st.markdown("### 📖 Scenario Briefing")
            st.markdown(f"**👤 Your Role:** {b.get('role', 'Salesperson')}")
            st.markdown(f"**📦 Product:** {b.get('product', 'N/A')}")
            st.markdown(f"**💰 Deal Value:** {b.get('deal_value', 'N/A')}")
            st.markdown(f"**📍 Scene:** {b.get('scene', 'N/A')}")
            st.markdown(f"**🎯 Objective:** {b.get('objective', 'N/A')}")
            st.markdown("---")
            st.markdown(f"**📝 Instructions:** {b.get('instructions', 'N/A')}")


# --- Main Interface ---

if st.session_state.analysis_loading:
    with st.spinner("🧠 Analyzing your performance with Gemini..."):
        transcript = ""
        for m in st.session_state.messages:
            if m["role"] == "user":
                transcript += f"Salesperson: {m['content']}\n"
            elif m["role"] == "assistant":
                transcript += f"Prospect: {m['content']}\n"
        
        st.session_state.assessment = asyncio.run(analyze_conversation(st.session_state.scenario_id, transcript))

    st.session_state.analysis_loading = False

if st.session_state.assessment:
    eval_data = st.session_state.assessment
    
    st.title("📊 Performance Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Overall Score")
        st.metric(label="", value=f"{eval_data.overall_score}/100")
        st.progress(eval_data.overall_score / 100)
        
        st.write("---")
        st.subheader("Tone & Style")
        ts = eval_data.speaking_tone_style
        st.write(f"Professional Tone: {ts.professional_tone}/10")
        st.progress(ts.professional_tone / 10)
        st.write(f"Active Listening: {ts.active_listening}/10")
        st.progress(ts.active_listening / 10)
        st.write(f"Engagement: {ts.engagement_quality}/10")
        st.progress(ts.engagement_quality / 10)

    with col2:
        st.subheader("Content Quality")
        cc = eval_data.conversation_content
        st.write(f"Needs Assessment: {cc.needs_assessment}/25")
        st.progress(cc.needs_assessment / 25)
        st.write(f"Value Proposition: {cc.value_proposition}/25")
        st.progress(cc.value_proposition / 25)
        st.write(f"Objection Handling: {cc.objection_handling}/20")
        st.progress(cc.objection_handling / 20)

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.success("✅ Strengths")
        for s in eval_data.strengths:
            st.write(f"- {s}")
    with c4:
        st.warning("⚠️ Improvements")
        for i in eval_data.improvements:
            st.write(f"- {i}")
            
    st.info(f"📝 **Coach's Feedback:** {eval_data.specific_feedback}")
    
    if st.button("🔙 Back to Chat"):
        st.session_state.assessment = None
        st.rerun()

else:
    # Chat Interface
    st.title(f"Roleplay: {scenarios[st.session_state.scenario_id]['name']}")
    
    scenario_data = scenarios[st.session_state.scenario_id]
    st.caption(scenario_data.get('description', ''))



    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Enter your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        client = get_gemini_client()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=st.session_state.messages
                )
                ai_response = response.choices[0].message.content
                st.write(ai_response)
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
