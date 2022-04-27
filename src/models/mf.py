


from keras.engine import training
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Input, Reshape, Dot
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.losses import mean_squared_error
from tensorflow import squeeze
import tensorflow as tf


# Define recommendation model
def RecommenderV1(n_users, n_movies, nu_factors, ni_factors, reg_param):
    # building user input
    user = Input(shape=(1,))
    user_embed = Embedding(n_users, nu_factors, embeddings_initializer='uniform',
                  embeddings_regularizer=l2(reg_param))(user)
    user_embed = Reshape((nu_factors,))(user_embed)
    
    # building movie input
    movie = Input(shape=(1,))
    movie_embed = Embedding(n_movies, ni_factors, embeddings_initializer='uniform',
                  embeddings_regularizer=l2(reg_param))(movie)
    movie_embed = Reshape((ni_factors,))(movie_embed)
    
    # putting together user and movie input
    dot_prod = Dot(axes=1)([user_embed, movie_embed])
    dot_prod_bias = Dense(1, use_bias=True, kernel_regularizer=l2(reg_param))(dot_prod)
    model = Model(inputs=[user, movie], outputs=dot_prod_bias)
    #opt = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer="adam")

    return model

def Recommender(n_users, n_movies, nu_factors, ni_factors, reg_param):
    # building user input
    user = Input(shape=(1,))
    user_embed = Embedding(n_users, nu_factors, embeddings_initializer='he_normal')(user)
    user_embed = Reshape((nu_factors,))(user_embed)
    
    # building movie input
    movie = Input(shape=(1,))
    movie_embed = Embedding(n_movies, ni_factors, embeddings_initializer='he_normal')(movie)
    movie_embed = Reshape((ni_factors,))(movie_embed)
    
    # putting together user and movie input
    dot_prod = Dot(axes=1)([user_embed, movie_embed])
    dot_prod_bias = Dense(1, activation='linear', use_bias=True)(dot_prod)
    model = Model(inputs=[user, movie], outputs=dot_prod_bias)
    loss = mean_squared_error()

    loss += reg_param * l2(squeeze(model.trainable_variables))
    print(loss)
    #opt = Adam(lr=0.001)
    model.compile(loss=loss, optimizer="adam")

    return model


if __name__ == '__main__':
    n_users = 100
    n_movies = 100
    n_factors = 10
    model = RecommenderV1(n_users, n_movies, n_factors, n_factors, 0.001)
    model.summary()
    for trainable_var in model.trainable_variables:

        print(trainable_var.shape)
    
        

    