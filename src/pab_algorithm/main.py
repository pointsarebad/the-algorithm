from pab_algorithm.predictor.predictor import ScorePredictor

predictor = ScorePredictor(
    model_type="linear",
    default_samples=1_000,
    use_cache=True,
    use_tilting=True,
    alpha=-0.4,
)

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
        except IndexError:
            print("Not a team")
        print("\n")
