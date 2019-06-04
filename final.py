import numpy as np
import pandas as pd
import datetime
from datetime import date
from datetime import datetime
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import seaborn as sns
import scipy
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn                        import metrics, svm
from sklearn.linear_model           import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
import warnings
warnings.filterwarnings('ignore')

class fit_ensemble:

    def __init__(self, data):
        self.data = data[['mean', 'heure_depart_min']].dropna(how='any', axis = 0)

    def imputer(self, data):
        # Convert the DataFrame object into NumPy array otherwise you will not be able to impute
        values = self.data.values

        # Now impute it
        imputer = Imputer()
        imputedData = imputer.fit_transform(values)
        self.data=imputedData

    def split(self, method):        
        self.method = method        
        self.X = np.array(self.data[:, 1])
        self.Y = np.array(self.data[:, 0], dtype = 'int')
        self.X = self.X.reshape(len(self.X),1)
        self.Y = self.Y.reshape(len(self.Y),1)

        if method == 'all':
            self.X_train = self.X
            self.Y_train = self.Y
            self.X_test  = self.X
            self.Y_test  = self.Y                        
        elif method == 'split':            
            self.X_train, self.X_test, self.Y_train, self.Y_test = \
                train_test_split(self.X, self.Y, test_size=0.3)
    
    def train(self):
        self.kfold = model_selection.KFold(n_splits=10, random_state=0)
        cart = DecisionTreeClassifier()
        num_trees = 100
        self.model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=0)
        
    
    def calc_score(self):
        clf = LogisticRegression()
        lab_enc = preprocessing.LabelEncoder()
        self.Y_train = lab_enc.fit_transform(self.Y_train)
        result = cross_val_predict(self.model, self.X_train, self.Y_train, cv = self.kfold)
        score = metrics.mean_squared_error(result, self.Y_train)
        print('Ecart = {:.2f} min'.format(np.sqrt(score)))

def get_flight_delays(df, carrier, id_airport, extrem_values = False):
    df2 = df[(df['AIRLINE'] == carrier) & (df['ORIGIN_AIRPORT'] == id_airport)]
    #_______________________________________
    # remove extreme values before fitting
    if extrem_values:
        df2['DEPARTURE_DELAY'] = df2['DEPARTURE_DELAY'].apply(lambda x:x if x < 60 else np.nan)
        df2.dropna(how = 'any')
    #__________________________________
    # Conversion: date + heure -> heure
    df2.sort_values('SCHEDULED_DEPARTURE', inplace = True)
    df2['heure_depart'] =  df2['SCHEDULED_DEPARTURE'].apply(lambda x:datetime.strptime(x, '%Y-%m-%d %H:%M:%S').time())
    #___________________________________________________________________
    # regroupement des vols par heure de départ et calcul de la moyenne
    test2 = df2['DEPARTURE_DELAY'].groupby(df2['heure_depart']).apply(get_stats).unstack()
    test2.reset_index(inplace=True)
    #___________________________________
    # conversion de l'heure en secondes
    fct = lambda x:x.hour*3600+x.minute*60+x.second
    test2.reset_index(inplace=True)
    test2['heure_depart_min'] = test2['heure_depart'].apply(fct)
    return test2

# function that extract statistical parameters from a grouby objet:
def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

df = pd.read_csv('_flights_.csv')
carrier = 'AA'
id_airport = 'BNA'
test2 = get_flight_delays(df, carrier, id_airport, True)

fit = fit_ensemble(test2)
fit.imputer(test2)
fit.split('all')
fit.train()
fit.calc_score()
