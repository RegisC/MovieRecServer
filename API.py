import flask
import recommender

app = flask.Flask(__name__)
engine = recommender.Engine()

app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def home():
	''' Point d'entrée de base '''
	return "<h1>Bonjour ami cinéphile</h1>"

@app.route('/recommend', methods=['GET'])
def make_rec():
	''' Point d'entrée de notre API '''
	req_args = flask.request.args
	if 'id' in req_args:
		print("Requête reçue avec paramètres :", req_args)
		id = req_args['id']
		n = req_args.get('n', default=5, type=int)
		sort_by = req_args.get('sort_by', default='dist')
		return engine.make_rec(id, n, sort_by)
	else:
		return "Erreur : aucun identifiant de film fourni."
		
# Lance le serveur
if __name__ == "__main__":
	app.run()
