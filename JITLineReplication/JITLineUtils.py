import re

python_common_tokens = ['abs','delattr','hash','memoryview','set','all','dict','help','min','setattr','any','dir','hex','next','slice','ascii','divmod','id','object','sorted','bin','enumerate','input','oct','staticmethod','bool','eval','int','open','str','breakpoint','exec','isinstance','ord','sum','bytearray','filter','issubclass','pow','super','bytes','float','iter','print','tuple','callable','format','len','property','type','chr','frozenset','list','range','vars','classmethod','getattr','locals','repr','zip','compile','globals','map','reversed','__import__','complex','hasattr','max','round','False','await','else','import','passNone','break','except','in','raise','True','class','finally','is','return','and','continue','for','lambda','try','as','def','from','nonlocal','while','assert','del','global','not','with','async','elif','if','or','yield', 'self']

def preprocess_code_line(code, remove_python_common_tokens=False):
    code = code.replace('(',' ').replace(')',' ').replace('{',' ').replace('}',' ').replace('[',' ').replace(']',' ').replace('.',' ').replace(':',' ').replace(';',' ').replace(',',' ').replace(' _ ', '_')
    code = re.sub('``.*``','<STR>',code)
    code = re.sub("'.*'",'<STR>',code)
    code = re.sub('".*"','<STR>',code)
    code = re.sub('\d+','<NUM>',code)
    
    if remove_python_common_tokens:
        new_code = ''

        for tok in code.split():
            if tok not in python_common_tokens:
                new_code = new_code + tok + ' '
            
        return new_code.strip()
    
    else:
        return code.strip()
    

def preprocess_diff(diff_content):    
    lines = str(diff_content).split("\n")
    removed_lines = []
    added_lines = []
    for line in lines:
        if line.startswith("@@"):
            continue  # Skip header lines
    
        if line.startswith("+"):
            line = line[line.index('+')+1:]
            if line.startswith("#"):
                continue
            hashtag_idx = line.find("#")
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            added_lines.append(preprocess_code_line(line,remove_python_common_tokens=True))
        elif line.startswith("-"):
            line = line[line.index('-')+1:]
            if line.startswith("#"):
                continue
            hashtag_idx = line.find("#")
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            removed_lines.append(preprocess_code_line(line,remove_python_common_tokens=True))

    added_code = ' \n '.join(list(set(added_lines)))
    removed_code = ' \n '.join(list(set(removed_lines)))
    return added_code + " " + removed_code