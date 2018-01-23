import sys
sys.path.append("..")
from config import *
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
from recurrentshop import *

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(
             (input_shape[-1],),
             initializer=self.init,
             name='{}_W'.format(self.name),
             regularizer=self.W_regularizer,
             constraint=self.W_constraint
        )
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(
                (input_shape[1],),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint
            )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f,kernel_size=c,padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_textcnn(seq_length, embed_weight):
    content = Input(shape=(seq_length,),dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0],weights=[embed_weight],output_dim=embed_weight.shape[1],trainable=False)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    feat = convs_block(trans_content)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2,activation="softmax")(fc)
    model = Model(inputs=content,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_hcnn(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32")
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    feat = convs_block(review_encode)
    dropfeat = Dropout(0.2)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2,activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_han(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32")
    embedding = Embedding(
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_embed = embedding(sentence_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_embed)
    word_attention = Attention(sent_length)(word_bigru)
    sent_encode = Model(sentence_input, word_attention)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    review_encode = TimeDistributed(sent_encode)(review_input)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(review_encode)
    sent_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(2,activation="softmax")(fc)
    model = Model(review_input, output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_word_char_cnn(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,),dtype="int32")
    char_input = Input(shape=(char_len,),dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block(trans_word, convs=[1,2,3,4,5], f = 256)
    char_feat = convs_block(trans_char, convs=[1,2,3,4,5], f = 256)
    feat = concatenate([word_input, char_feat])
    dropfeat = Dropout(0.4)(feat)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2,activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_wordp_han(sent_num, sent_length, embed_weight, mask_zero=False):
    sentence_input = Input(shape=(sent_length,), dtype="int32")
    sentencep_input = Input(shape=(sent_length,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=embed_weight.shape[0],
        weights=[embed_weight],
        output_dim=embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    wordp_embedding = Embedding(
        name="wordp_embedding",
        input_dim=57,
        output_dim=64
    )
    sent_word_encode = Model(sentence_input, word_embedding(sentence_input))
    sent_wordp_encode = Model(sentencep_input, wordp_embedding(sentencep_input))

    sent_union_input = Input(shape=(sent_length, embed_weight.shape[1]+64,))
    sent_union_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_union_input)
    sent_union_output = Attention(sent_length)(sent_union_bigru)
    sent_encode = Model(sent_union_input, sent_union_output)

    review_input = Input(shape=(sent_num, sent_length), dtype="int32")
    reviewp_input = Input(shape=(sent_num, sent_length), dtype="int32")
    sent_word_embed = TimeDistributed(sent_word_encode)(review_input)
    sent_wordp_embed = TimeDistributed(sent_wordp_encode)(review_input)
    sent_embed_union = concatenate([sent_word_embed, sent_wordp_embed])
    review_encode = TimeDistributed(sent_encode)(sent_embed_union)
    sent_bigru = Bidirectional(GRU(128, return_sequences=True))(review_encode)
    sent_attention = Attention(sent_num)(sent_bigru)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(sent_attention)))
    output = Dense(2,activation="softmax")(fc)
    model = Model([review_input, reviewp_input], output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_word_char_hcnn(sent_num, sent_word_length, sent_char_length, word_embed_weight, char_embed_weight, mask_zero=False):
    sentence_word_input = Input(shape=(sent_word_length,), dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_word_embed = word_embedding(sentence_word_input)
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(sent_word_embed)
    word_attention = Attention(sent_word_length)(word_bigru)
    sent_word_encode = Model(sentence_word_input, word_attention)

    sentence_char_input = Input(shape=(sent_char_length,), dtype="int32")
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        mask_zero=mask_zero,
        trainable=False
    )
    sent_char_embed = char_embedding(sentence_char_input)
    char_bigru = Bidirectional(GRU(64, return_sequences=True))(sent_char_embed)
    char_attention = Attention(sent_char_length)(char_bigru)
    sent_char_encode = Model(sentence_char_input, char_attention)

    review_word_input = Input(shape=(sent_num, sent_word_length), dtype="int32")
    review_word_encode = TimeDistributed(sent_word_encode)(review_word_input)
    review_char_input = Input(shape=(sent_num, sent_char_length), dtype="int32")
    review_char_encode = TimeDistributed(sent_char_encode)(review_char_input)
    review_encode = concatenate([review_word_encode, review_char_encode])
    unvec =  convs_block(review_encode, convs=[1,2,3,4,5], f = 256)
    dropfeat = Dropout(0.2)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(256)(dropfeat)))
    output = Dense(2,activation="softmax")(fc)
    model = Model([review_word_input, review_char_input], output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

### V2

def convs_block_v2(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Conv1D(f, c, activation='elu', padding='same')(data)
        conv = MaxPooling1D(c)(conv)
        conv = Conv1D(f, c-1, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(c-1)(conv)
        conv = Conv1D(f/2, 2, activation='elu', padding='same')(conv)
        conv = MaxPooling1D(2)(conv)
        conv = Flatten()(conv)
        pools.append(conv)
    return concatenate(pools, name=name)

def get_textcnn_v2(seq_length, embed_weight):
    content = Input(shape=(seq_length,),dtype="int32")
    embedding = Embedding(input_dim=embed_weight.shape[0],weights=[embed_weight],output_dim=embed_weight.shape[1],name="embedding",trainable=False)
    trans_content = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
    unvec = convs_block(trans_content)
    dropfeat = Dropout(0.4)(unvec)
    fc = Dropout(0.4)(Activation(activation="relu")(BatchNormalization()(Dense(300)(dropfeat))))
    output = Dense(2,activation="softmax")(fc)
    model = Model(inputs=content,outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_word_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,),dtype="int32")
    char_input = Input(shape=(char_len,),dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(word_embedding(word_input)))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))

    word_feat = convs_block_v2(trans_word)
    char_feat = convs_block_v2(trans_char)

    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dropout(0.2)(Dense(512)(dropfeat))))
    output = Dense(2,activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model

def get_wordp_char_cnn_v2(word_len, char_len, word_embed_weight, char_embed_weight):
    word_input = Input(shape=(word_len,),dtype="int32")
    wordp_input = Input(shape=(word_len,),dtype="int32")
    char_input = Input(shape=(char_len,),dtype="int32")
    word_embedding = Embedding(
        name="word_embedding",
        input_dim=word_embed_weight.shape[0],
        weights=[word_embed_weight],
        output_dim=word_embed_weight.shape[1],
        trainable=False
    )
    wordp_embedding = Embedding(
        name="wordp_embedding",
        input_dim=57,
        output_dim=64
    )
    char_embedding = Embedding(
        name="char_embedding",
        input_dim=char_embed_weight.shape[0],
        weights=[char_embed_weight],
        output_dim=char_embed_weight.shape[1],
        trainable=False
    )
    word_union = concatenate([word_embedding(word_input), wordp_embedding(wordp_input)])
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(320))(word_union))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(char_embedding(char_input)))))
    word_feat = convs_block_v2(trans_word)
    char_feat = convs_block_v2(trans_char)
    unvec = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(unvec)
    fc = Activation(activation="relu")(BatchNormalization()(Dense(512)(dropfeat)))
    output = Dense(2,activation="softmax")(Dropout(0.2)(fc))
    model = Model(inputs=[word_input, wordp_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    return model
