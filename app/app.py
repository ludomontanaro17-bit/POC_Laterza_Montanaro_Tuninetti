from API import API

if __name__ == "__main__":
    api = API(
        model_paths="../model/keras_model.h5",
        scaler_path="../model/scaler.pkl"
    )
    api.run()

