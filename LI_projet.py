from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import pickle


db = pd.read_csv("Preprocessing/data.csv", delimiter=";")
sentences = db['Texte']
languages = db['Langue']
language = set(db.Langue.values)

# constitution des sous-corpus
x_train, x_test, y_train, y_test = train_test_split(sentences, languages, random_state=42, test_size=0.1)

# instanciation des outils de vectorisation
tf_idf = TfidfVectorizer(ngram_range=(3, 4), analyzer='char', max_features=3000, lowercase=True)
cv = CountVectorizer(ngram_range=(3, 4), analyzer='char', max_features=3000, lowercase=True)


def precision_rappel_fmes(prediction, test_class):
    """
    Calcule la précision, le rappel et la f-mesure pour chaque langue
    :param prediction : les prédictions faites par le classifieur
    :param test_class : les valeurs cibles
    """

    pre_rap = {}

    list_precision = []
    list_rappel = []

    for i, element in enumerate(test_class):
        if prediction[i] == element:
            pre_rap[element] = pre_rap.get(element, 0) + 1 # nombre de fois que la langue d'une phrase a bien été prédite
    # calcul de la précision
    for classe in pre_rap:
        score = pre_rap[classe] / len([x for x in prediction if x == classe])
        list_precision.append(score)
        print(f"précision {classe}: {score}")

    print()
    # calcul du rappel
    for classe in pre_rap:
        score = pre_rap[classe] / len([x for x in test_class if x == classe])
        list_rappel.append(score)
        print(f"rappel {classe}: {score}")

    print()
    # calcul de la f-mesure
    for classe, precision, rappel in zip(pre_rap, list_precision, list_rappel):
        f_mesure = 2*precision*rappel / (precision+rappel)
        print(f"f-mesure {classe}: {f_mesure}")


def matrix_confusion(y_test, y_pred, name):
    """
    Créée la matrice de confusion à partir des prédictions du classifieur et des valeurs cibles
    :param y_test : valeurs cibles
    :param y_pred : valeur prédite
    """

    cm = confusion_matrix(y_target=y_test, y_predicted=y_pred) # instanciation d'un objet matrice

    fig, ax = plt.subplots(figsize=(15, 10)) # fixe la taille
    ax.set_title(name)
    plot_confusion_matrix(conf_mat=cm, cmap=plt.cm.YlGn, class_names=set(y_test), axis=ax) # création de la matrice
    plt.show() # affichage de la matrice


def lrt():

    print("Logistic Regression avec TFIDF")

    model = pickle.load(open("model_lr-tfidf", "rb"))

    lrt_pred = model.predict(x_test) # prédiction des phrases du corpus de test

    accuracy = accuracy_score(y_test, lrt_pred) # calcul de l'exactitude des prédictions du modèle

    print(f"Accuracy is : {accuracy}")
    precision_rappel_fmes(lrt_pred, y_test) # calcul précision, rappel, f-mesure
    matrix_confusion(y_test, lrt_pred, "Logistic Regression avec TFIDF") # création matrice de confusion

    do_test = input("Voulez faire un test avec ce modèle ? y/n ")
    if do_test.lower() == "y":
        yn = True
        while yn:
            sent_to_test = str(input("Entrez une phrase dans une langue de votre choix : "))
            print(model.predict([sent_to_test]))
            yn = input("Voulez-vous tester une autre phrase ? y/n ")
            if yn.lower() == "n":
                yn = False
            else:
                continue
    else:
        pass


def lrcv():

    print("Logistic Regression avec CountVectorizer")

    model = pickle.load(open("model_lr-cv", "rb"))

    lrcv_pred = model.predict(x_test) # prédiction des phrases du corpus de test

    accuracy = accuracy_score(y_test, lrcv_pred) # calcul de l'exactitude des prédictions du modèle
    print(f"Accuracy is : {accuracy}")

    precision_rappel_fmes(lrcv_pred, y_test) # calcul précision, rappel, f-mesure
    matrix_confusion(y_test, lrcv_pred, "Logistic Regression avec CountVectorizer") # création matrice de confusion

    do_test = input("Voulez faire un test avec ce modèle ? y/n ")
    if do_test.lower() == "y":
        yn = True
        while yn:
            sent_to_test = str(input("Entrez une phrase dans une langue de votre choix : "))
            print(model.predict([sent_to_test]))
            yn = input("Voulez-vous tester une autre phrase ? y/n ")
            if yn.lower() == "n":
                yn = False
            else:
                continue
    else:
        pass


def nbt():

    print("Naive Bayes avec TFIDF")

    model = pickle.load(open("model_nb-tfidf", "rb"))
    tf_idf.fit_transform(x_train) # création matrice terme-document + apprentissage vocabulaire
    x_test_tfidf = tf_idf.transform(x_test) # transformation du corpus de test en matrice terme-document

    y_pred_tfidf = model.predict(x_test_tfidf.toarray()) # génération des prédictions

    accuracy = accuracy_score(y_test, y_pred_tfidf) # calcul de l'exactitude des prédictions du modèle
    print(f"Accuracy is : {accuracy}")

    precision_rappel_fmes(y_pred_tfidf, y_test) # calcul précision, rappel, f-mesure
    matrix_confusion(y_test, y_pred_tfidf, "Naive Bayes avec TFIDF") # création matrice de confusion

    do_test = input("Voulez faire un test avec ce modèle ? y/n ")
    if do_test.lower() == "y":
        yn = True
        while yn:
            sent_to_test = str(input("Entrez une phrase dans une langue de votre choix : "))
            print(model.predict(tf_idf.transform([sent_to_test]).toarray()))
            yn = input("Voulez-vous tester une autre phrase ? y/n ")
            if yn.lower() == "n":
                yn = False
            else:
                continue
    else:
        pass


def nbcv():

    print("Naive Bayes avec CountVectorizer")

    model = pickle.load(open("model_nb-cv", "rb"))
    cv.fit_transform(x_train) # création matrice terme-document + apprentissage vocabulaire

    x_test_cv = cv.transform(x_test)  # transformation du corpus de test en matrice terme-document
    y_pred_cv = model.predict(x_test_cv.toarray()) # génération des prédictions

    accuracy = accuracy_score(y_test, y_pred_cv)# calcul de l'exactitude des prédictions du modèle
    print(f"Accuracy is : {accuracy}")

    precision_rappel_fmes(y_pred_cv, y_test) # calcul précision, rappel, f-mesure
    matrix_confusion(y_test, y_pred_cv, "Naive Bayes avec CountVectorizer") # création matrice de confusion

    do_test = input("Voulez faire un test avec ce modèle ? y/n ")
    if do_test.lower() == "y":
        yn = True
        while yn:
            sent_to_test = str(input("Entrez une phrase dans une langue de votre choix : "))
            print(model.predict(cv.transform([sent_to_test]).toarray()))
            yn = input("Voulez-vous tester une autre phrase ? y/n ")
            if yn.lower() == "n":
                yn = False
            else:
                continue
    else:
        pass


if __name__ == '__main__':

    lrt()
    lrcv()
    nbt()
    nbcv()
