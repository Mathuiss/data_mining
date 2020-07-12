# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


# %%
# Loading data into dataframe
df = pd.read_csv("data/planets.csv")
print(df.head())


# %%
# The dataset contains planet data, and a label which indicated whether or not it is a candidate to be a habitable planet.
# If it is a candidate, it needs to be manually reviewed, and will be either confirmed, or labeled a false positive.
# Our goal is to use data science techniques to autimate the review process (or attempt to).

# Steps to take:
# - Preprocessing: We must make sure that we only have cases where the review has been done. This way we can make sure that the AI is trained on accurate data.
# - Preprocessing: We must normalize the data, so that we are not comparing apples to oranges.
# - Preprocessing: We must label the data, so that on our y axis, we can classify between habitable and unhabitable (0 or 1).
# - Model building: We must build different models, using different classification techniques, to determine the bitability of each planet.
# - Training: We must train different models, and adjust training according to the evaluation of the training.
# - Evaluating: We will measure the performance and training of each model, and adjust accordingly.
# - Presentation: We will try to visualize the results in such a way, that a human can easily understand it.


# %%
# Preprocessing: Removing irrelevant data

print(df.columns)
columns_to_drop = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_tce_delivname"]
df = df.drop(columns=columns_to_drop, axis=1)

# Removing CANDIDATE DATA, becaus we only want the planets that are checked
df = df[df["koi_disposition"] != "CANDIDATE"]


# %%
# Preprocessing: Normalizing the data
print(df.columns)
columns_to_normalize = ['koi_fpflag_nt',
                        'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_period',
                        'koi_period_err1', 'koi_period_err2', 'koi_time0bk', 'koi_time0bk_err1',
                        'koi_time0bk_err2', 'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
                        'koi_duration', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth',
                        'koi_depth_err1', 'koi_depth_err2', 'koi_prad', 'koi_prad_err1',
                        'koi_prad_err2', 'koi_teq', 'koi_teq_err1', 'koi_teq_err2', 'koi_insol',
                        'koi_insol_err1', 'koi_insol_err2', 'koi_model_snr', 'koi_tce_plnt_num', 'koi_steff', 'koi_steff_err1', 'koi_steff_err2',
                        'koi_slogg', 'koi_slogg_err1', 'koi_slogg_err2', 'koi_srad',
                        'koi_srad_err1', 'koi_srad_err2', 'ra', 'dec', 'koi_kepmag']

df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# %%
# Dropping unused column, because we are already hot-one-encoding the pdisposition field, and have removed all the CANDIDATE's
df = df.drop(columns=["koi_disposition", "koi_score", "koi_teq_err1", "koi_teq_err2"], axis=1)
df = df.fillna(0)
# We will also hot-one-encode the koi_disposition column, because we can then compare the score to the encoding, and determine the certainty.
for i, r in df.iterrows():
    if r["koi_pdisposition"] == "CANDIDATE":
        df.at[i, "koi_pdisposition"] = 1
    else:
        df.at[i, "koi_pdisposition"] = 0


# %%
# Preprocessing: Labeling the data
# We split the data into tnto x and y (features and lables)
y_cols = ["koi_pdisposition"]
y = df[y_cols]
x = df.drop(columns=y_cols, axis=1)


# %%
# Here we split the x and y into training and test datasets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=2020)


# %%
# this function takes a classifier and all the data, it will then use that classifyer to perform a classification on the planets.
# Afterwards it will print how many were correct, and how accuracte it was.
def perform_measure(classifyer, x_train, y_train, x_test, y_test):
    classifyer = classifyer.fit(x_train, y_train)
    y_pred = classifyer.predict(x_test)

    y_test = y_test.values.reshape((-1,))

    correct = 0

    for i in range(len(y_test)):
        pred = y_pred[i]
        actual = y_test[i]

        if pred == actual:
            correct += 1

    print(f"Correct: {correct}")
    print(f"Total: {len(y_test)}")
    accuracy = correct / len(y_test)
    print(f"Accuracy: {accuracy}")
    return accuracy


metrics = []

metrics.append(perform_measure(DummyClassifier(strategy="most_frequent"), x_train, y_train, x_test, y_test))
metrics.append(perform_measure(DummyClassifier(strategy="stratified"), x_train, y_train, x_test, y_test))
metrics.append(perform_measure(DummyClassifier(strategy="prior"), x_train, y_train, x_test, y_test))
metrics.append(perform_measure(DummyClassifier(strategy="uniform"), x_train, y_train, x_test, y_test))


# %%
# Here we can see how well each classifier performs
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
types = ["Most Frequent", "Stratisfied", "Priot", "Uniform"]
ax.bar(types, metrics)
plt.show()


# %%
# In the above image we can see how accurate we are in predicting the habitability of planets.
# We see each type of classigication method an the different results.


# %%
# In order to work with keras we need to transform our data into numpy arrays


print(type(y_train))

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(y_train)

np.save("models/datasets/planets_x_train", x_train)
np.save("models/datasets/planets_x_test", x_test)
np.save("models/datasets/planets_y_train", y_train)
np.save("models/datasets/planets_y_test", y_test)


# %%
# Load dataset from hard drive
x_train = np.load("models/datasets/planets_x_train.npy")
x_test = np.load("models/datasets/planets_x_test.npy")
y_test = np.load("models/datasets/planets_y_test.npy")
y_train = np.load("models/datasets/planets_y_train.npy")


# We create a new sequential model
model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(40,)))
model.add(Dense(1, activation="sigmoid"))

# Compiling the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

arr_metrics = model.fit(x_train, y_train, batch_size=32, epochs=6, validation_data=(x_test, y_test))

model.summary()

evaluation = model.evaluate(x_test, y_test)
print("Validation Loss, Validation Accuracy")
print(evaluation)

print(arr_metrics.history.keys())
plt.plot(arr_metrics.history["accuracy"])
plt.plot(arr_metrics.history["val_accuracy"])
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"])
plt.show()

plt.plot(arr_metrics.history["loss"])
plt.plot(arr_metrics.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Traing", "Val"])
plt.show()


# %%
