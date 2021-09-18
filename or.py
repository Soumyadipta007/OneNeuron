from utils.model import Perceptron
from utils.all_utils import prepare_data,save_model,save_plot
import pandas as pd
import  logging
import os


logging_str="[%(asctime)s: %(levelname)s: %(module)s:] %(message)s"
logging.basicConfig(level=logging.INFO, format=logging_str)

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=os.path.join(log_dir,"current_logs.log"),level=logging.INFO,format=logging_str)

def main(data,eta,epochs,filename,plotFilename):
    df = pd.DataFrame(data)
    logging.info(f"This is a dataframe{df}")

    X,y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model,filename=filename)
    save_plot(df, file_name=plotFilename, model=model)

if __name__ == "__main__":
    OR = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,1,1,1],
    }

    ETA = 0.3 # between 0 and 1
    EPOCHS = 10
    filename="or.model"
    plotFilename="or.png"
    try:
        logging.info(">>> starting the training")
        main(data=OR,eta=ETA,epochs=EPOCHS,filename= filename, plotFilename=plotFilename)
        logging.info(">>> stopped the training")
    except Exception as e:
        logging.exception(e)
        raise  e