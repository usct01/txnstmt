import pandas as pd
pd.set_option('mode.chained_assignment', None)

import re
from unicodedata import normalize as unicode_normalize
from abydos.distance import JaroWinkler
JW=JaroWinkler()


def remove_suffix(astring,suffix):
    astring=str(astring)
    if astring.endswith(suffix):
        end = len(suffix)
        return astring[0:-end]
    return astring

abbrev_dict={'dubbed':'dub','season ':'s '}

def clean_name(phrase: str) -> str:
    mypunc='!"#$%&\'()*+,-:;<=>?@[\\]^_`{|}~/'
    phrase=str(phrase)
    if phrase is not None and len(phrase)>1:
        phrase = re.sub('&', ' and ', phrase)
        phrase = re.sub("[\(\[].*?[\)\]]", "", phrase)
        phrase = unicode_normalize('NFKD', phrase.strip().lower())
        # phrase = remove_suffix(phrase,' the')
        phrase = phrase.translate(str.maketrans(mypunc, ' '*len(mypunc)))
        phrase = ''.join([c for c in phrase if c.isalnum() or c.isspace()])
        # phrase = " ".join([abbrev_dict.get(x,x) for x in phrase.split()])
        for k,v in abbrev_dict.items():
            if ' ' in k:
                phrase=phrase.replace(k, v)
        phrase = " ".join([abbrev_dict.get(x,x) for x in phrase.split()])
        return phrase
    else:
        return phrase    


def match_key(name,prefix='the'):
    name=str(name)
    if name is not None and type(name) == str and len(name)>1:
        name=clean_name(name)
        name=name[name.startswith(prefix) and len(prefix):].strip()
        name=remove_suffix(name,' the')
        # name=re.sub(grp_pat, "", name).strip()
        # for suffix in comp_stop_words:
        #     name=remove_suffix(name,suffix)
        return ''.join(sorted(name.split()))
#             return ' '.join(sorted(set(name.split())))
    else:
        return name
def mname_sim_dask(ldt,rdt,lkey,rkey):
    ldt['N1W']=ldt[lkey].str.split(n=1).str.get(0)
    gudict=dict(zip(ldt.N1W.tolist(),ldt[lkey].tolist()))
    gudict={k: v for k, v in gudict.items() if len(k) >2}
    mlst=[]  
    for gu,val in gudict.items():
        if len(gu)>2:
            srch=rdt[rdt[rkey].str.startswith(gu)]
#             srch=rdt[rdt[rkey].str.contains(gu)]
            if len(srch.index) > 0:
                srch[lkey]=val
                mlst.append(srch)

    mdt1=pd.concat(mlst)
    mdt1=mdt1.assign(NAME_SIMILARITY_SCORE=mdt1[[lkey,rkey]].apply(lambda x: 0 if any(x.isnull()) else JW.sim(str(x[0]),str(x[1])), axis=1))
    mdt1.sort_values([lkey,'NAME_SIMILARITY_SCORE'],ascending=[True,False],inplace=True)
    mdt1.drop_duplicates(subset=[lkey],inplace=True)
    return mdt1