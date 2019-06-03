from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model



def nn_print_plot(model, out_file_name="model_plot.png"):
    """
    :param model: neural network
    :param out_file_name: string
    :return:
    """

    print(model.summary())
    plot_model(model, to_file=out_file_name, show_shapes=True, show_layer_names=True)


def nn_model_test(number_outputs_1, number_outputs_2):
    """
    :param number_inputs_0: int
    :param number_inputs_1: int
    :return: keras nn model
    """
    model = Sequential()
    model.add(Dense(number_outputs_1, input_dim=5, activation='relu'))
    model.add(Dense(number_outputs_2, activation='sigmoid'))
    return model


