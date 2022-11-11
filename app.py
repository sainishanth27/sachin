import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from flask import Flask ,render_template,request

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
data_frame.drop(["mean texture", "mean smoothness", "mean symmetry", "mean fractal dimension", "texture error", "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst texture", "worst smoothness", "worst symmetry", "worst fractal dimension" ], axis = 1, inplace=True)
data_frame['label'] = breast_cancer_dataset.target
# data_frame.isnull().sum()

data_frame['label'].value_counts()
data_frame.groupby('label').mean()
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = GaussianNB()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

app=Flask(__name__)


@app.route("/")
def maya():
    return render_template("index.html")

@app.route("/predict")
def m():
    return render_template("predict.html")

@app.route("/age")
def mayukha():
    return render_template("age.html")

@app.route("/ivsd")
def mayukhaa():
    return render_template("ivsd.html")

@app.route("/sevVsNonsev")
def mayukhaaa():
    return render_template("sevVsNonsev.html")

@app.route("/pairplot")
def mayukhaaaa():
    return render_template("pairplot.html")

@app.route("/index")
def ma():
    return render_template("index.html")

@app.route("/team")
def may():
    return render_template("team.html")

@app.route("/diagnosis")
def mayu():
    return render_template("diagnosis.html")

@app.route("/features")
def mayuu():
    return render_template("features.html")

@app.route("/submit", methods=['GET','POST'])

def submit():
    a=float(request.form.get('mean_radius'))
    # b=float(request.form.get('mean_texture'))
    c=float(request.form.get('mean_perimeter'))
    d=float(request.form.get('mean_area'))
    # e=float(request.form.get('mean_smoothness'))
    f=float(request.form.get('mean_compactness'))
    g=float(request.form.get('mean_concavity'))
    h=float(request.form.get('mean_concave_points'))
    # i=float(request.form.get('mean_symmetry'))
    # j=float(request.form.get('mean_fractal_dimension'))
    k=float(request.form.get('radius_error'))
    # l=float(request.form.get('texture_error'))
    m=float(request.form.get('perimeter_error'))
    n=float(request.form.get('area_error'))
    # o=float(request.form.get('smoothness_error'))
    # p=float(request.form.get('compactness_error'))
    # q=float(request.form.get('concavity_error'))
    # r=float(request.form.get('concave_points_error'))
    # s=float(request.form.get('symmetry_error'))
    # t=float(request.form.get('fractal_dimension_error'))
    u=float(request.form.get('worst_radius'))
    # v=float(request.form.get('worst_texture'))
    w=float(request.form.get('worst_perimeter'))
    x=float(request.form.get('worst_area'))
    # y=float(request.form.get('worst_smoothness'))
    z=float(request.form.get('worst_compactness'))
    aa=float(request.form.get('worst_concavity'))
    ab=float(request.form.get('worst_concave_points'))
    # ac=float(request.form.get('worst_symmetry'))
    # ad=float(request.form.get('worst_fractal_dimension'))

    input_data = (a,c,d,f,g,h,k,m,n,u,w,x,z,aa,ab)

    input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return render_template("predict.html",p='Report : The Breast Cancer is Severe!')

    else:
        return render_template("predict.html",p='Report : The Breast Cancer is Not Severe!')


if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run( debug =True)