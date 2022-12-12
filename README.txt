TU Justine
PITER Kenza
BAUNÉ Julie

README - DÉTECTION DE LA LANGUE


Ce programme vise à identifier la langue d'une phrase donnée en input\
et alternativement à mesurer à l'aide de différentes métriques, les performances de 2 méthodes 
de classifications paramétrées de différentes manières.

Ce programme a été développé avec python 3.10 sur MacOS 12.1 et a été testé avec succès 
avec python 3.9 sur MacOS 12.1 et python 3.6.9 sur Ubuntu 18.04.

Les packages nécessaires à l'exécution du programme sont listés dans le fichier requirements.txt

Le programme permet de détecter la langue d'une phrase, ou plus généralement 
d'une chaine de caractères donnée en input par l'utilisateur. Il fournira également les métriques de performances de chacune des méthodes de classifications pour chaque paramétrage.

Le notebook "model_trainings-LD.ipynb" permet d'entraîner chacun des modèles sur le jeu de donnée afin de mettre en oeuvre la détection. 
Chaque modèle peut être directement chargé depuis les 4 fichiers model.

Le programme ne nécessite pas d'argument pour être lancé. Il suffit d'exécuter la commande suivante dans le terminal : 
python3 projet_LI.py
