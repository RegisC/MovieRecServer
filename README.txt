DESCRIPTION

API de recommandation de films : étant donnée un film choisi par l'utilisateur, 
l'API renvoie une liste de films similaires.

UTILISATION

http://rc-movie-rec.herokuapp.com/recommend?id=<id>&n=<n>&sort_by=<tri>

Paramètre requis 

id : identifiant IMDb du film

Paramètres optionnels 

n : nombre de films à renvoyer 
sort_by : trier les résultats par similarité ("dist") ou note ("imdb_score" )

EXEMPLE

http://rc-movie-rec.herokuapp.com/recommend?id=tt4263482&n=10&sort_by=imdb_score

DÉPLOIEMENT

git push heroku master

REMARQUE CONCERNANT LES PACKAGES

Les 2 packages mkl-fft et mkl-random qui ne sont pas utilisés mais installés 
comme dépendances doivent être manuellement retirés de `requirements.txt` car
ils peuvent poser problème à Heroku.