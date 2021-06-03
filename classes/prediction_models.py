import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'classes'))

from classes import data_manager

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPool1D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import autokeras as ak

from xgboost import XGBRegressor
from xgboost import XGBClassifier

# Prediction models: 
# regressors and classifiers that take a positinal embedding vector as input, 
# output storypoints (or other impact related value)


seed=42

class regressors:

    @staticmethod
    def get_autokeras_paraphrase5():
        model = load_model("regression_models/autokeras5_desc_paraphrase_rmse", custom_objects=ak.CUSTOM_OBJECTS)
        return model

    @staticmethod
    def get_autokeras_roberta3_mae():
        model = load_model("regression_models/autokeras3_roberta_mae", custom_objects=ak.CUSTOM_OBJECTS)
        return model

    @staticmethod
    def keras_convolutional(X_train, y_train, X_test, y_test, vocab_size, max_len):
        #https://realpython.com/python-keras-text-classification/
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, mode='min')
        ]

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size+1,
                            output_dim=50,
                            input_length=max_len))
        model.add(Conv1D(50, 5, activation='relu'))
        model.add(GlobalMaxPool1D())
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=1, activation='relu'))

        model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='mae',
                    metrics=['mse'],
                    run_eagerly=True)

        history = model.fit(X_train, y_train,
                            epochs=15,
                            verbose=True,
                            validation_data=(X_test, y_test),
                            batch_size=50,
                            callbacks=callbacks)

        return model

    @staticmethod
    def create_MLP(X, y):
        model = MLPRegressor(random_state=seed)
        model = model.fit(X, y)

        pipe = Pipeline([('mlp', model)])
        param_grid = {
            'mlp__solver': ['sgd'],
            'mlp__alpha': [0.01],
            'mlp__learning_rate_init': [0.0001],
            'mlp__max_iter': [300]
        }

        gs = gridsearch(pipe, param_grid, 'neg_mean_squared_error')
        gs.fit(X, y)

        data_manager.print_gridsearch_best_stats(gs)
        return model

    @staticmethod
    def create_SVR(X, y):
        model = svm.SVR()

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('svr', model)])
        param_grid = {
            'svr__C': [1.75], #1.957,1.8,2,1.7 #multi lang 1.75
            'svr__gamma': ['scale'],
            'svr__kernel': ['rbf'],
            'svr__epsilon': [0.01], #0.1,0.01 #multi lang 0.01
            'svr__degree': [2] #2,3,4
        }

        gs = gridsearch(pipe, param_grid, 'neg_mean_absolute_error')  #neg_mean_squared_error
        gs.fit(X, y)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod
    def create_Randomforest(X, y):
        model = RandomForestRegressor(random_state=seed, n_estimators=300, min_samples_leaf=4, max_depth=20)

        pipe = Pipeline([('rtree', model)])

        param_grid = {
            # 'rtree__n_estimators': [300],
            # 'rtree__min_samples_leaf': [4],
            # 'rtree__max_depth': [20]
        }

        gs = gridsearch(pipe, param_grid, 'neg_mean_absolute_error')
        gs.fit(X, y)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod
    def create_XGBregressor(X_train, y_train):
        model = XGBRegressor(learning_rate=0.001,
                            n_estimators=400,
                            n_jobs=5,
                            random_state=seed)

        pipe = Pipeline([('XGB', model)])
        param_grid = {
        }

        gs = gridsearch(pipe, param_grid, 'neg_mean_squared_error')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)
        return model

    @staticmethod
    def keras_sequential_network(X_train, y_train, X_test, y_test, lr=0.001):
        input_dim = len(X_train[0])

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, mode='min')
        ]

        model = Sequential()
        model.add(Dense(100, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mse', 'mae'], run_eagerly=True)

        model.fit(X_train, y_train,
                            epochs=15,
                            verbose=True,
                            validation_data=(X_test, y_test),
                            batch_size=50,
                            callbacks=callbacks)

        pipe = Pipeline([('nn', model)])

        param_grid = {}

        gs = gridsearch(pipe, param_grid, 'neg_mean_squared_error')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return model


class classifiers(object):
    @staticmethod #0.7133 - F 0.7220 - H 0.7281 - H2 0.73152
    def create_mlpclassifier(X_train, y_train):
        model = MLPClassifier(random_state=seed)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('sgd', model)])

        param_grid = {
            'mlp__max_iter':[200], #200, 400, 600, 800 | 200
            'mlp__solver':['adam'], #'adam', 'lbfgs' | 'adam'
            'mlp__alpha':[0.001], #0.0001, 0.001 | 0.001
            'mlp__batch_size':[50], #100, 150, 200, 400 | 50
            'mlp__learning_rate_init':[0.0001] #0.01, 0.001, 0.0001 | 0.0001
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.6709 - F 0.6817
    def create_Randomforest(X_train, y_train):
        model = RandomForestClassifier(random_state=seed)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('sgd', model)])

        param_grid = {
            # 'rtree__n_estimators': [700], #best from range 150 - 700
            # 'rtree__min_samples_leaf': [2], #best from range 1 - 7
            # 'rtree__max_depth': [20]
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.7147 - F 0.7206 - H 0.7256 - H2 0.72656
    def create_XGB(X_train, y_train):
        model = XGBClassifier(seed=seed, 
                              use_label_encoder=False)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('xgb', model)])

        param_grid = {
            'xgb__learning_rate':[0.05, 0.03], #0.2, 0.1, 0.15, 0.01 | 0.05
            'xgb__n_estimators':[600, 800], #100, 300, 400, 500 | 600
            'xgb__max_depth':[7], #4, 5, 6, 7, 8 | 7
            'xgb__colsample_bytree':[0.2], #0.1, 0.2 | 0.2
            'xgb__reg_lambda':[4, 6, 8] #1, 2, 3, 4 | 4
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.6750 - F 0.6885
    def create_GB(X_train, y_train):

        #max_depth=6, n_estimators=500, random_state=42))])
        # best parms:  {'gb__learning_rate': 0.1, 'gb__max_depth': 6, 'gb__n_estimators': 500}
        model = GradientBoostingClassifier(random_state=seed)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('gb', model)]) 
        param_grid = {
            # 'gb__n_estimators': [500], #50 - 600
            # 'gb__learning_rate': [0.1], #0.2 - 0.01
            # 'gb__max_depth': [6], #1-7
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.7152 - F 0.7195 H 0.73417
    def create_SVC(X_train, y_train):
        model = svm.SVC(random_state=seed,
                        probability=True)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('svc', model)])

        param_grid = {
            'svc__kernel': ['rbf'], #'rbf', 'linear' | rbf
            'svc__degree': [2], #2,3,4 | 2
            'svc__gamma': ['scale'], #'auto', 'scale' | 'scale'
            'svc__C': [1.95] #1, 1.95 | 1.95
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.6670 - F 0.6735
    def create_KNN(X_train, y_train):
        model = KNeighborsClassifier()

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('KNN', model)])

        param_grid = {
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #0.6764 - F 0.667
    def create_SGD(X_train, y_train):
        model = SGDClassifier(random_state=seed)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('sgd', model)])

        param_grid = {
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod #F - 0.7311 - H2 0.73587
    def create_voting(X_train, y_train):
        SVC = svm.SVC(random_state=seed,
                      probability=True,
                      kernel='rbf',
                      degree=2,
                      gamma='scale',
                      C=1.95)
        XGB = XGBClassifier(seed=seed,
                            learning_rate=0.05,
                            n_estimators=600,
                            max_depth=7,
                            reg_lambda=4,
                            colsample_bytree=0.2,
                            use_label_encoder=False)
        MLP = MLPClassifier(random_state=seed, 
                            max_iter=200,
                            solver='adam',
                            alpha=0.001,
                            batch_size=50,
                            learning_rate_init=0.0001)

        estimators = [
            ('svc', SVC),
            ('xgb', XGB),
            ('mlp', MLP)
        ]

        model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[1,1,1],
            n_jobs=-1,
            verbose=True)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('vc', model)])

        param_grid = {
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        print('voting done')
        return gs

    @staticmethod #F - 0.72848 - H2 0.7373
    def create_stacking(X_train, y_train):
        SVC = svm.SVC(random_state=seed,
                      probability=True,
                      kernel='rbf',
                      degree=2,
                      gamma='scale',
                      C=1.95)
        XGB = XGBClassifier(seed=seed,
                            learning_rate=0.05,
                            n_estimators=600,
                            max_depth=7,
                            reg_lambda=4,
                            colsample_bytree=0.2,
                            use_label_encoder=False)
        MLP = MLPClassifier(random_state=seed, 
                            max_iter=200,
                            solver='adam',
                            alpha=0.001,
                            batch_size=50,
                            learning_rate_init=0.0001)

        estimators = [
            ('svc', SVC),
            ('xgb', XGB),
            ('mlp', MLP)
        ]

        model = StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(random_state=42)
        )

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('stack', model)])

        param_grid = {
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        print('stacking done')

        data_manager.print_gridsearch_best_stats(gs)

        return gs

    @staticmethod
    def create_logisticregression(X_train, y_train):
        model = LogisticRegression(random_state=42)

        pipe = Pipeline([('standardize', StandardScaler()), 
                         ('lg', model)])
        param_grid = {
            'lg__max_iter':[600]
        }

        gs = gridsearch(pipe, param_grid, 'recall_macro')
        gs.fit(X_train, y_train)

        data_manager.print_gridsearch_best_stats(gs)

        return gs


def gridsearch(pipe, param_grid, metric):
    gs = GridSearchCV(pipe,
                      param_grid,
                      verbose=0,
                      cv=5,
                      scoring=metric,
                      n_jobs=4,
                      return_train_score=True)

    return gs
