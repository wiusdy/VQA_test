from transformers import pipeline

clf = pipeline("sentiment-analysis")
output = clf("Man, I love how bad the patriots are doing this year!!")
print(output)