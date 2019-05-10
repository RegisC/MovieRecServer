import flask
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

class Engine: 
	__shared_state = {} # simule un 'singleton'
	DIM_PCA = 150
	NUM_DIRECTORS = 30
	NUM_COUNTRIES = 12
	NUM_CLUSTERS = 80
	CALIBRATE_LOCALLY = False
	
	def __init__(self):
		self.__dict__ = self.__shared_state # technique du BORG
		self.__load_data()
		if Engine.CALIBRATE_LOCALLY:
			self.__create_clusters()
		else:
			self.__load_categories()
		self.__init_columns_for_distance_calc()

	def __load_data(self):
		source = 'compact-imdb.csv'
		self.data = pd.read_csv(source, delimiter='\t', keep_default_na=False)
		print(f"DataFrame chargé de dimensions {self.data.shape}")
		print(f"Taille en mémoire : {self.data.memory_usage(True, True).sum()}")
		self.data.set_index('id', inplace=True)
			
	def __dist_between_keywords(self, id1, id2):
		kw1 = set(self.data.loc[id1].plot_keywords.split('|'))
		kw2 = set(self.data.loc[id2].plot_keywords.split('|'))
		intersection = kw1.intersection(kw2)
		n = min(len(kw1), len(kw2))
		return (n - len(intersection)) / n		

	def __load_categories(self):
		"""Charge les catégories calibrées au préalable."""
		source = 'categories.csv'
		self.df_categories = pd.read_csv(source, delimiter='\t')
		self.df_categories.set_index('id', inplace=True)
		
	def __create_clusters(self):
		"""Calibre le modèle de partitionnement à partir des données brutes."""
		# création du tableau de données pour partionnement
		cols = [col for col in self.data if col.startswith('genre_')]
		cols.extend(['adj_content_rating', 'adj_title_year', 'adj_imdb_score'])
		X = self.data[cols]
		# ajout des colonnes de mots-clefs
		fv = TfidfVectorizer(tokenizer=lambda s: s.split("|"))
		plot_kw_freq = fv.fit_transform(self.data['plot_keywords']).todense()
		plot_kw_freq_list = Engine.__get_feature_list(fv, 'kw_')
		# suppression de la colonne d'absence de mot-clef
		key_to_delete = 'kw_'
		if key_to_delete in plot_kw_freq_list:
			pos = plot_kw_freq_list.index(key_to_delete)
			plot_kw_freq = np.delete(plot_kw_freq, pos, axis=1)
			plot_kw_freq_list.remove(key_to_delete)
		# ACP sur les mots-clefs
		pca = PCA(n_components=Engine.DIM_PCA)
		plot_kw_pca = pca.fit_transform(plot_kw_freq);	
		pca_cols = [ 'ACP' + str(c) for c in list(range(0, Engine.DIM_PCA))]
		df_kw = pd.DataFrame(plot_kw_pca, columns=pca_cols, index=X.index)
		X = pd.concat([X, df_kw], axis=1)
		print(f"Taille du tableau de données après ACP : {X.shape}")
		# Ajout des réalisateurs et des pays
		cv = CountVectorizer(tokenizer=lambda s: s.split("|"), 
							 max_features=Engine.NUM_DIRECTORS)
		directors = cv.fit_transform(self.data['director_name']).todense()
		dir_list = Engine.__get_feature_list(cv, 'dir_')
		df_directors = pd.DataFrame(directors, columns=dir_list, index=X.index)
		X = pd.concat([X, df_directors], axis=1)
		print(f"Taille du tableau de données avec réalisateurs : {X.shape}")		
		cv = CountVectorizer(tokenizer=lambda s: s.split("|"), 
							 max_features=Engine.NUM_COUNTRIES)
		countries = cv.fit_transform(self.data['country']).todense()
		c_list = Engine.__get_feature_list(cv, 'country_')
		df_countries = pd.DataFrame(countries, columns=c_list, index=X.index)
		X = pd.concat([X, df_countries], axis=1)
		print(f"Taille du tableau de données avec pays : {X.shape}")
		# Calibration du modèle
		model = KMeans(n_clusters=Engine.NUM_CLUSTERS, random_state=42)
		self.df_categories = pd.DataFrame(model.fit_predict(X), 
										  index=self.data.index, 
										  columns=['category'])		
		print(f"Taille du tableau de catégories : {self.df_categories.shape}")

	def __make_recs(self, id, n, sort_by, verbose):
		cat = self.df_categories.loc[id, 'category']
		mask = (self.df_categories['category'] == cat) & (self.data.index != id)
		shortlist = self.data.loc[mask, ['movie_title', 'imdb_score']]
		title = self.data.loc[id, 'movie_title']
		print(f"Taille de la partition de '{title}' : {len(shortlist)+1}")
		# Calcul de la distance
		dist = lambda movie: self.calc_dist(movie, id, 'cosine')
		shortlist['dist'] = shortlist.index.to_series().apply(dist)
		# Tri des résultats 
		sort_asc = False if sort_by == 'imdb_score' else True
		shortlist.sort_values(by=sort_by, ascending=sort_asc, inplace=True)
		shortlist.rename(columns={'movie_title': 'name'}, inplace=True)		
		if not(verbose):
			shortlist.drop(columns=['dist', 'imdb_score'], inplace=True)
		return shortlist.iloc[0:n, :]

	def __init_columns_for_distance_calc(self):
		# colonnes utilisées pour le calcul de distance
		self.numeric_cols = [c for c in self.data if c.startswith('genre_')] 
		self.numeric_cols.extend(Engine.__get_numeric_cols_for_distance())
		self.cat_cols = ['director_name', 'country', 'language']
		self.set_cols = ['actor_1_name', 'actor_2_name', 'actor_3_name']
		self.keyword_col = 'plot_keywords'
		
	def calc_dist(self, id1, id2, metric):
		"""Calcule la distance entre les 2 films spécifiés."""
		film1 = self.data.loc[id1]
		film2 = self.data.loc[id2]
		n = len(self.numeric_cols) + len(self.cat_cols) + 2
		v1 = np.zeros(n).reshape(1, n)
		v2 = np.zeros(n).reshape(1, n)
		# Copie des valeurs des attributs numériques
		for i, col in enumerate(self.numeric_cols):
			v1[0, i] = film1[col]
			v2[0, i] = film2[col]
		# Test d'égalité des attributs catégoriels
		for i, c in enumerate(self.cat_cols):
			v2[0, len(self.numeric_cols)+i] = 0 if film1[c] == film2[c] else 1    
		# Calcul du nb d'éléments communs aux 2 ensembles
		set1 = { film1[col] for col in self.set_cols }
		set2 = { film2[col] for col in self.set_cols }
		v2[0, n-2] = (len(set1) - len(set1.intersection(set2))) / len(set1)
		# Calcul du nombre de mots-clefs communs
		v2[0, n-1] = self.__dist_between_keywords(id1, id2)
		# Calcul de la distance entre les 2 vecteurs
		return pairwise_distances(v1, v2, metric=metric)[0][0]		
		
	def make_rec(self, id, n, sort_by, verbose=False):
		"""Renvoie `n` recommandations de films similaires au film `id`"""
		if (id in self.data.index):
			df = self.__make_recs(id, n, sort_by, verbose).reset_index(level=0)
			dict = {'_results': df.to_dict(orient='records')}
		else:
			dict = {'_error': 'unknown movie id'}		
		return flask.jsonify(dict)
	
	@staticmethod
	def __get_numeric_cols_for_distance():
		"""Renvoie la liste des colonnes contenant des valeurs numériques."""
		cols = ['adj_budget', 'adj_cast_total_facebook_likes', 
				'adj_director_facebook_likes', 'adj_duration', 
				'adj_imdb_score', 'adj_title_year', 'adj_num_voted_users',
				'adj_content_rating']
		return cols
	
	@staticmethod
	def __get_feature_list(vectorizer, prefix):
		"""Retourne la liste des attributs extraits de l'objet `vectorizer`."""
		return [prefix + kw.replace(' ', '_') 
				for kw in vectorizer.get_feature_names()]	
				
if __name__ == "__main__":
	"""Test unitaire simple exécuté en mode script."""
	app = flask.Flask(__name__)
	with app.app_context():
		engine = Engine()
		# Recommandations pour Snow White
		res = engine.make_rec('tt0029583', 5, 'dist', verbose=True)
		print(res.response)
		res = engine.make_rec('tt0029583', 5, 'imdb_score', verbose=True)
		print(res.response)
		