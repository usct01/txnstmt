import re
import pandas as pd
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer


def process_data(data):
    data = data['data'] # keep it small for now
    return data

def json_extract(obj, key):
    """Recursively fetch values from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values

def json_to_df(data):
    """Extract id and description from transactions"""
    id = json_extract(data, key="id")
    description = json_extract(data, key="description")

    df = pd.DataFrame(list(zip(id, description)),
              columns=['id','description'])

    # df["id"] = pd.to_numeric(df["id"])

    return df

def preprocess(description):
    description = str(description)
    description = description.replace('\n', ' ')
    description = description.lower()
    description = re.sub('[0-9]+', '', description)
    description = re.sub('[^[:alnum:]]', ' ', description)
    description = description.strip()
    
    return(description)

def stem_description(data):
    data['stem_description']= data['description'].map(lambda s:preprocess(s))

    return(data) 

def bert_embeddings(data):
    txtlst = data['stem_description'].tolist()
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(txtlst, show_progress_bar=True)
    embeddings_reduced = umap.UMAP(n_components=5, metric='cosine').fit_transform(embeddings)
    embeddings_reduced = pd.DataFrame(data=embeddings_reduced, 
        columns=["embed_1", "embed_2", "embed_3", "embed_4", "embed_5"])

    data = data.drop('stem_description', axis=1)
    data = data.drop('description', axis=1)
    output = pd.concat([data, embeddings_reduced], axis=1)

    return(output)

def lambda_handler(event, context):
	data = process_data(event)
	data = json_to_df(data)
	data = stem_description(data)
	output = bert_embeddings(data)
	output = output.to_dict(orient='records')

	return jsonify({"Embeddings": output})



