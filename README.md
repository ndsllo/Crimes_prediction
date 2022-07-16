## Crimes Prediction in Montgomery County in the State of Maryland
Auteurs: BALL Alhousseynou, JOURDAN Fanny, DIOP Seyni, LO Ndeye, SIDIBE Bakary


Le comté de Montgomery est l’un des 23 comtés de l'État du Maryland aux États-Unis. Situé au nord-ouest de Washington, il est constitué de 46 villes et son siège se trouve à Rockville. C'est le comté le plus peuplé de l'Etat et sa population est en constante augmentation. C'est d'ailleurs l'une zone les plus riches des Etats-Unis. Malgré une politique qui cherche à réduire le nombre de crime et délit dans le comté à l'image de l'application du contrôle important sur la vente d'alcool, il a enregistré plus de 40 000 crimes en 2018.

Il serait alors une question de sécurité publique de pouvoir prédire le nombre de crimes par semaine suivant chaque ville. Cette information serait d’une utilité capitale dans la mesure où elle permettrait de mieux identifier les zones à risque et ainsi renforcer la présence de la police dans ces zones.

#### Installation

1. Installer ramp-workflow library
```
!pip install https://api.github.com/repos/paris-saclay-cds/ramp-workflow/zipball/master
```
2. Suivre les instructions de [ramp-kit](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit) 

3. Installer folium (pas obligatoire, ça permet juste de visualiser le comté de Montgomery)
```
!pip install folium 
```
4. Pour les utilisateurs de google colab, cloner le dossier githup dans drive
```
from google.colab import drive
drive.mount('/content/drive')
%cd "/content/drive/My Drive"
!git clone https://github.com/balldatascientist/Crimes_prediction.git
%cd "/content/drive/My Drive/Crimes_prediction"
```
