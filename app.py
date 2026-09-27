from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("cricket_win_probability_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get values entered by the user
        runs_left = float(request.form["runs_left"])
        balls_left = float(request.form["balls_left"])
        wickets = float(request.form["wickets"])
        target_runs = float(request.form["target_runs"])

        # Calculate current run rate
        runs_scored = target_runs - runs_left
        balls_played = 120 - balls_left

        if balls_played > 0:
            cur_run_rate = (runs_scored / balls_played) * 6
        else:
            cur_run_rate = 0

        # Calculate required run rate
        if balls_left > 0:
            req_run_rate = (runs_left / balls_left) * 6
        else:
            req_run_rate = 0

        # Prepare input for the ML model
        new_match = [[
            runs_left,
            balls_left,
            wickets,
            target_runs,
            cur_run_rate,
            req_run_rate
        ]]

        # Get probabilities
        probability = model.predict_proba(new_match)

        win_probability = probability[0][1] * 100
        loss_probability = probability[0][0] * 100

        return render_template(
            "index.html",
            win_probability=round(win_probability, 2),
            loss_probability=round(loss_probability, 2),
            current_run_rate=round(cur_run_rate, 2),
            required_run_rate=round(req_run_rate, 2)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)