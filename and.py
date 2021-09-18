from os import name
from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import pandas as pd
import logging

logging_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s:]"
logging.basicConfig(level=logging.INFO, format=logging_str)

def main(data,eta,epochs,filename,plotFilename):
    df = pd.DataFrame(data)
    logging.info(f"This is the given dataframe{df}")

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename=filename)
    save_plot(df, file_name=plotFilename, model=model)

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3 # between 0 and 1
    EPOCHS = 10
    filename="and.model"
    plotFilename="and.png"
    main(data=AND,eta=ETA,epochs=EPOCHS,filename= filename, plotFilename=plotFilename)
