import streamlit as st
import json
from main_agent import LangGraphRecruitingAgent

# Set page config
st.set_page_config(page_title="LangGraph Recruiting Agent", layout="wide")

# App header
st.title("🔍 LangGraph Recruiting Agent")
st.markdown("Automatically evaluate and rank candidates using LLM + LangGraph workflow.")

# Instantiate the agent
@st.cache_resource
def get_agent():
    return LangGraphRecruitingAgent()

agent = get_agent()

# Job description input
st.subheader("📄 Enter Job Description")
job_description = st.text_area("Paste the job description below:", height=300)

if st.button("🚀 Evaluate Candidates"):
    if not job_description.strip():
        st.warning("Please enter a valid job description.")
    else:
        with st.spinner("Running LangGraph Evaluation... ⏳"):
            results = agent.run_evaluation(job_description)

        if "error" in results:
            st.error(f"Error: {results['error']}")
        else:
            st.success(f"✅ Evaluation completed. {results['total_candidates_evaluated']} candidates evaluated.")
            
            # Display Assessment Criteria
            with st.expander("📋 Assessment Criteria"):
                st.json(results["assessment_criteria"])

            # Display Top 3 Candidates
            st.subheader("🏆 Top 3 Candidates")

            for candidate in results["top_candidates"]:
                st.markdown(f"### 🥇 Rank {candidate['rank']}: {candidate['name']}")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Score", f"{candidate['total_score']:.1f}%")
                    st.write("**Technical Skills:**", f"{candidate['skill_score']:.1f}%")
                    st.write("**Experience:**", f"{candidate['experience_score']:.1f}%")
                    st.write("**Education:**", f"{candidate['education_score']:.1f}%")
                    st.write("**Alignment:**", f"{candidate['alignment_score']:.1f}%")

                with col2:
                    st.write("**Address:**", candidate["contact_info"].get("address", "N/A"))
                    st.write("**Matched Skills:**", ", ".join(candidate["detailed_evaluation"]["matched_skills"]))
                    st.write("**Experience Years:**", candidate["detailed_evaluation"]["experience_years"])

                with st.expander("📧 Outreach Email"):
                    st.markdown(candidate["outreach_email"])

                with st.expander("🧠 Decision Explanation"):
                    st.markdown(candidate["decision_explanation"])

                st.markdown("---")
