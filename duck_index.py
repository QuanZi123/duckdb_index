import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

data = []
k=4
with open("train.jsonl", 'r') as file:
    for line in file:
        data.append(json.loads(line))
df = pd.DataFrame(data)

queries = ['表盘小，表带短']
passages = df["text"].values
print(len(passages))

model = SentenceTransformer('BAAI/bge-small-zh')

p_embeddings = model.encode(passages, normalize_embeddings=True)
q_embeddings = model.encode([q for q in queries], normalize_embeddings=True)

table_data = []
for i in range(len(p_embeddings)):
    table_data.append([passages[i],p_embeddings[i]])

print("---frist-create-index---")
conn1 = duckdb.connect("speed_3.db")
conn1.execute("""
install vss
""")


conn1.execute("""
load vss
""")

conn1.execute("""
set hnsw_enable_experimental_persistence = true
""")

conn1.execute(f"""
CREATE TABLE  IF NOT EXISTS  my_trans_vec (text VARCHAR,vec FLOAT[{p_embeddings.shape[1]}])
""")


conn1.execute("""
CREATE INDEX  my_trans_hnsw_index ON my_trans_vec USING HNSW (vec) WITH (metric = 'cosine')
""")

conn1.executemany("""
INSERT INTO my_trans_vec (text ,vec) values (?,?)
""",table_data)


res1 = conn1.execute(f"""
    SELECT text FROM my_trans_vec ORDER BY array_cosine_distance(vec, ?::FLOAT[{p_embeddings.shape[1]}]) LIMIT {k};
    """,q_embeddings).fetchall()
print(res1)



print("---frist-insert-data---")
conn2 = duckdb.connect("speed_t.db")
conn2.execute("""
install vss
""")

conn2.execute("""
load vss
""")

conn2.execute("""
set hnsw_enable_experimental_persistence = true
""")

conn2.execute(f"""
CREATE TABLE  IF NOT EXISTS  my_trans_vec (text VARCHAR,vec FLOAT[{p_embeddings.shape[1]}])
""")

conn2.executemany("""
INSERT INTO my_trans_vec (text ,vec) values (?,?)
""",table_data)

conn2.execute("""
CREATE INDEX  my_trans_hnsw_index ON my_trans_vec USING HNSW (vec) WITH (metric = 'cosine')
""")
res = conn2.execute(f"""
    SELECT text FROM my_trans_vec ORDER BY array_cosine_distance(vec, ?::FLOAT[{p_embeddings.shape[1]}]) LIMIT {k};
    """,q_embeddings).fetchall()
print(res)


