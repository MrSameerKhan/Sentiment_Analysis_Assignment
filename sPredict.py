import tensorflow as tf
from keras.preprocessing import sequence
import pickle


class predict:

    """
    This is an sentiment analysis using text classification
    """

    def predictModel(self,inputText):

        loaded_Model = tf.keras.models.load_model('./weights/sentiment.hs')

        with open('./weights/sentiment.pickle', 'rb') as handle:
            loaded_tokenizer = pickle.load(handle)

        sequence_T= loaded_tokenizer.texts_to_sequences([inputText])
        padded = sequence.pad_sequences(sequence_T, maxlen=64)
        prediction = loaded_Model.predict_classes(padded)

        labelOutput = "Negative" if prediction == [0] else "Positive" if prediction == [1] else None

        return labelOutput


if __name__ == "__main__":

    stringInput = """Not RecommendedUnplayable ever since the Free-to-Play update. If you think the cheating issue was a problem before, it just got worse. It is impossible to have a single match without atleast a cheater in it. Unless Valve takes action, expect a lot of actual players permanently signing off. Do yourselves a favour and start off the new year by not playing this game.R.I.P CS:GO 2012-2018. """
    sPredict = predict()
    sOutput = sPredict.predictModel(stringInput)
    print("The review of the game seems to be : ", sOutput)
    print(predict.__doc__)
