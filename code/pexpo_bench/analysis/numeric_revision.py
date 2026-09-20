"""Explicit rounding and multi-metric scoring for author-corrected items.

Missing components receive zero. Rounding never changes a small positive risk to
zero. All scalar references and rounding policies are fixed before reading answers.
"""
import math
import re

_NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
_SUP=str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺','0123456789-+')

def number(text):
    if isinstance(text,(float,int)):return float(text)
    s=str(text or '').replace('−','-')
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=re.sub(r'10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)',lambda m:'10^'+m[1].translate(_SUP),s)
    s=re.sub(r'('+_NUM+r')\s*[×x*·]\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?',lambda m:f'{m[1]}e{m[2]}',s)
    m=re.search(_NUM,s)
    return float(m[0]) if m else None

def rounding_allowance(reference,policy):
    if not policy or reference==0:return 0.0
    if policy['kind']=='decimal_places':
        allowance=.5*10**(-int(policy['places']))
        # A policy that would round a positive scientific value to zero is invalid.
        return allowance if abs(reference)>=allowance*2 else 0.0
    if policy['kind']=='significant_figures':
        return .5*10**(math.floor(math.log10(abs(reference)))-int(policy['digits'])+1)
    raise ValueError('Unknown rounding policy')

def default_rounding(reference,unit=''):
    """Apply the same author-requested precision convention to every numeric item."""
    if reference is None or reference==0:return None
    if abs(reference)<.01:return {'kind':'significant_figures','digits':3}
    counts=bool(re.search(r'death|case|person|people',str(unit),re.I))
    return {'kind':'decimal_places','places':0 if counts else 2}

def scalar_score(pred,reference,tol,policy=None):
    from .scoring import calc_score
    if pred is None or reference is None or not math.isfinite(pred):return 0.0
    base=calc_score(pred,reference,tol)
    allowance=rounding_allowance(reference,policy)
    if allowance and abs(pred-reference)<=allowance+abs(reference)*1e-12:return 1.0
    return base

def component_predictions(answer,names):
    if isinstance(answer,dict):
        folded={str(k).casefold():v for k,v in answer.items()}
        return {k:number(folded.get(k.casefold())) for k in names}
    text=str(answer or '')
    result={}
    for name in names:
        # Read only the submitted answer field, never reasoning or question inputs.
        m=re.search(r'\b'+re.escape(name)+r'\b\s*(?:\([^\n]*?\))?\s*(?:=|:|≈|is)?\s*([^\n;,]+)',text,re.I)
        result[name]=number(m[1]) if m else None
        if m and '%' in m[1] and result[name] is not None:result[name]/=100
    if all(v is None for v in result.values()):
        # The prompt explicitly specifies order; accept an unlabelled two-value list.
        parts=re.split(r'[,;\n]',text.strip().strip('[]()'))
        if len(parts)==len(names) and all(re.fullmatch(r'\s*'+_NUM+r'\s*',p) for p in parts):
            result=dict(zip(names,map(number,parts)))
        elif len(names)==2 and re.fullmatch(r'\s*'+_NUM+r'\s*',text):
            result[names[0]]=number(text)  # Only the first component was supplied.
    return result

def named_component_score(answer,spec,tol):
    components=spec['components']
    pred=component_predictions(answer,list(components))
    scores={name:scalar_score(pred[name],item['value'],tol,item.get('rounding')) for name,item in components.items()}
    assert spec['aggregation']=='mean'
    return sum(scores.values())/len(scores),{'predictions':pred,'scores':scores}
