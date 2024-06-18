from pab_algorithm.predictor.predictor import ScorePredictor

predictor = ScorePredictor(model_type="linear", default_samples=1_000, use_cache=True)

# home = "england"
# away = "spain"

# print(predictor.get_win_probs(home, away))
# print(predictor.display_score(home, away))

if __name__ == "__main__":
    while True:
        home = input("Home: ")
        if home == "exit":
            break

        away = input("Away: ")
        if away == "exit":
            break

        try:
            print(predictor.display_score(home, away))
        except:
            print("Not a team")
        print("\n")
