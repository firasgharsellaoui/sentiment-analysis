import pickle


loaded_model_ara = pickle.load(open('sentiment_analysis_model_ara.sav', 'rb'))
loaded_model_tun = pickle.load(open('sentiment_analysis_model_tun.sav', 'rb'))
bow_model_ara = pickle.load(open('bow_model_ara.sav', 'rb'))
bow_model_tun = pickle.load(open('bow_model_tun.sav', 'rb'))


def predict_sentiment_ara(text):
    inst=[]
    inst.append(text)
    text_vect=bow_model_ara.transform(inst)
    prob=loaded_model_ara.predict_proba(text_vect)[0]
    x=prob[0]
    print(x)
    if x <0.7 and x>0.3:
        return("NEU")
    else:
        return loaded_model_ara.predict(text_vect)[0]

def predict_sentiment_tun(text):
    inst=[]
    inst.append(text)
    text_vect=bow_model_tun.transform(inst)
    prob=loaded_model_tun.predict_proba(text_vect)[0]
    x=prob[0]
    print(x)
    if x <0.8 and x>0.2:
        return("NEU")
    else:
        return loaded_model_tun.predict(text_vect)[0]
		
print(predict_sentiment_tun("زبالة لابارك الله"))