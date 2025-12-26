import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IPL Predictor", layout="wide")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    inning1_model = joblib.load("Innings1.pkl")
    inning2_model = joblib.load("IPL_model.pkl")
    return inning1_model, inning2_model

inning1_pipe, inning2_pipe = load_models()

# ---------------- CONSTANTS ----------------
teams = [
    'Chennai Super Kings','Delhi Capitals','Gujarat Titans',
    'Kolkata Knight Riders','Lucknow Super Giants','Mumbai Indians',
    'Punjab Kings','Rajasthan Royals','Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

cities = [
    'Ahmedabad','Bengaluru','Chennai','Delhi','Hyderabad',
    'Kolkata','Lucknow','Mumbai','Pune','Jaipur','Dharamsala'
]

# ---------------- UI ----------------
st.title("🏏 IPL Match Prediction System")

tab1, tab2 = st.tabs([
    "1️⃣ First Innings Score Predictor",
    "2️⃣ Second Innings Winner Predictor"
])

# =========================================================
# TAB 1 : FIRST INNINGS SCORE
# =========================================================
with tab1:
    st.header("First Innings Score Prediction")

    col1, col2 = st.columns(2)

    with col1:
        bat_team = st.selectbox("Batting Team", teams)
        bowl_team = st.selectbox("Bowling Team", teams)

    with col2:
        toss_winner = st.selectbox("Toss Winner", teams)
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
        current_score = st.number_input("Current Score", 0, step=1)
        wickets_fallen = st.number_input("Wickets Fallen", 0, 10, step=1, key="wickets1")


    st.markdown("### Overs Completed")
    over_col, ball_col = st.columns(2)

    with over_col:
        overs = st.number_input("Overs", 0, 20, step=1)

    with ball_col:
        balls = st.selectbox("Balls", [0, 1, 2, 3, 4, 5])

    if st.button("Predict Final Score"):
        if bat_team == bowl_team:
            st.error("Batting and Bowling teams must be different")
        else:
            balls_bowled = overs * 6 + balls
            balls_left = max(0, 120 - balls_bowled)
            overs_left = balls_left / 6
            overs_completed = overs + balls / 10
            crr = current_score / overs_completed if overs_completed > 0 else 0

            input_df = pd.DataFrame({
                "batting_team": [bat_team],
                "bowling_team": [bowl_team],
                "toss_winner": [toss_winner],
                "toss_decision": [toss_decision],
                "Overs_left": [overs_left],
                "balls_left": [balls_left],
                "current_score": [current_score],
                "wickets": [wickets_fallen],
                "CRR": [crr]
            })

            score = int(inning1_pipe.predict(input_df)[0])
            st.success(f"🏏 Predicted Final Score: **{score}**")
            st.write(f"Estimated Range: **{score-10} – {score+10}**")

# =========================================================
# TAB 2 : SECOND INNINGS WINNER
# =========================================================
with tab2:
    st.header("Second Innings Winner Prediction")

    col3, col4 = st.columns(2)

    with col3:
        chase_team = st.selectbox("Batting Team (Chasing)", teams)
        defend_team = st.selectbox("Bowling Team (Defending)", teams)
        city2 = st.selectbox("City", cities)

    with col4:
        target = st.number_input("Target Score", 1, step=1)
        score_now = st.number_input("Current Score", 0, step=1, key="score2")
        wickets_fallen2 = st.number_input("Wickets Fallen", 0, 10, step=1, key="wickets2")


    st.markdown("### Overs Completed")
    over_col2, ball_col2 = st.columns(2)

    with over_col2:
        overs2 = st.number_input("Overs", 0, 20, step=1, key="overs2")

    with ball_col2:
        balls2 = st.selectbox("Balls", [0, 1, 2, 3, 4, 5], key="balls2")

    if st.button("Predict Winner"):
        if chase_team == defend_team:
            st.error("Teams must be different")
        elif score_now >= target:
            st.success(f"🎉 {chase_team} already won!")
        else:
            balls_bowled2 = overs2 * 6 + balls2
            balls_left2 = max(1, 120 - balls_bowled2)
            runs_left = target - score_now
            wickets_left = 10 - wickets_fallen2

            overs_completed2 = overs2 + balls2 / 10
            crr2 = score_now / overs_completed2 if overs_completed2 > 0 else 0
            rrr = (runs_left * 6) / balls_left2

            input_df2 = pd.DataFrame({
                "batting_team": [chase_team],
                "bowling_team": [defend_team],
                "city": [city2],
                "runs_left": [runs_left],
                "balls_left": [balls_left2],
                "wickets": [wickets_left],
                "runs_target": [target],
                "CRR": [crr2],
                "RRR": [rrr]
            })

            probs = inning2_pipe.predict_proba(input_df2)[0]
            win_prob = probs[1] * 100
            loss_prob = probs[0] * 100

            st.subheader("Winning Probability")
            c1, c2 = st.columns(2)
            c1.metric(chase_team, f"{win_prob:.2f}%")
            c2.metric(defend_team, f"{loss_prob:.2f}%")

            if win_prob > loss_prob:
                st.success(f"✅ {chase_team} likely to win")
            else:
                st.error(f"❌ {defend_team} likely to win")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Developed by **Ayush Chauhan** | IPL ML Prediction System")

