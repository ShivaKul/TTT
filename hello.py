import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def hello_world():
	if request.method == "POST":
		issue_number = request.form.get("issue_number")
		issue_number = int(issue_number)

		corpus = ['Expected payment file isn\'t available for approval',
		'Problem uploading payment file',
		'Request assistance with file import issue Payment file error not uploading file successfully.',
		'Unable to send Bacs payments as certificate error',
		'Can we copy a payment file into a new system']

		threshold_value= 0.7

		tfidf_vectorizer = TfidfVectorizer()

		tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

		cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

		print(cosine_sim)

		scores = dict()

		for ix,iy in np.ndindex(cosine_sim.shape):
			if ix < iy and cosine_sim[ix,iy]>= threshold_value:
				scores[ix,iy]= cosine_sim[ix,iy]

		sorted_scores = sorted(scores.items(), reverse = True, key=lambda item: item[1])

		matching_issues = ["issue description1", "issue description2", "issue description3"]

		return render_template('index.html', matching_issues=matching_issues)
	else:
		return render_template('index.html', matching_issues=None)