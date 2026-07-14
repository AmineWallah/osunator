from tensorflow import keras
from tensorflow.keras import layers
from osunator.mdn import mdn_nll, MDN_PARAMS

N_INPUT_FEATURES = 9
N_CURSOR_OUTPUTS = 2
N_KEY_OUTPUTS = 4

BATCH_SIZE = 1 # raise it after making custom training loop

def build_model(lstm_units=(256, 128, 64)):
    inputs = keras.Input(batch_shape=(BATCH_SIZE, None, N_INPUT_FEATURES), name='inputs')
    x = inputs
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(units, return_sequences=True, stateful=True, name=f"lstm_{i}")(x)

    cursor_out = layers.Dense(MDN_PARAMS, name='cursor_out')(x)  # NO activation — raw mixture params
    key_out = layers.Dense(N_KEY_OUTPUTS, activation='sigmoid', name='key_out')(x)

    model = keras.Model(inputs=inputs, outputs=[cursor_out, key_out], name='osunator_lstm')
    return model

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={
            'cursor_out': mdn_nll,
            'key_out': 'binary_crossentropy',
        },
        loss_weights={'cursor_out': 1.0, 'key_out': 0.5}, # edit later on
    )
    return model

if __name__ == '__main__':
    model = compile_model(build_model())
    model.summary()