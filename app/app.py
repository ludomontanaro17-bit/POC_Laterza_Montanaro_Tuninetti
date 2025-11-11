from API import API

if __name__ == "__main__":
    api = API(
        model_path="../model/keras_model.h5",
        scaler_path="../model/scaler.pkl"
    )
    api.run()

