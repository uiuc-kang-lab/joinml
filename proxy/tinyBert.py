from sentence_transformers.cross_encoder import CrossEncoder
import time
model = CrossEncoder("tinyBertQQPCrossEncoder")
start = time.time()
score = model.predict(["Can we ever store energy produced in lightning?", 
                       "Is it possible to store the energy of lightning?"])
end = time.time()
print(score)
print(end-start)