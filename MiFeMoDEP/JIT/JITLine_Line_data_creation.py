import pandas as pd
import numpy as np
import re, time

path_to_jit_random = './Datasets/defectors/jit_bug_prediction_splits/random'

train_df = pd.read_parquet(f'{path_to_jit_random}/train.parquet.gzip')
train_df = train_df.reset_index(drop=True)

test_df = pd.read_parquet(f'{path_to_jit_random}/test.parquet.gzip')
test_df = test_df.reset_index(drop=True)

test_np = np.array(test_df["lines"][0])
# observation there is an extra "\n" in content of diff accordings to line numbers of header
def is_not_empty(str):
    if(len(str) == 0):
        return False
    else:
        return True

def remove_empty(li):
    return list(filter(is_not_empty,li))

def diff_header_process(header):
    dict = {}
    list = remove_empty(header.split("@@"))
    
    nums = remove_empty(list[0].split(" "))
    
    old = remove_empty(nums[0].split("-"))[0]
    old = remove_empty(old.split(","))
    if(len(old)<2):
        dict_old = { 'start': int(old[0]), 'num': int(old[0])}
    else:
        dict_old = { 'start': int(old[0]), 'num': int(old[1])}
    dict['old'] = dict_old
    new = remove_empty(nums[1].split("+"))[0]
    new = remove_empty(new.split(","))
    if(len(new)<2):
        dict_new = { 'start': int(new[0]), 'num': int(new[0])}
    else:
        dict_new = { 'start': int(new[0]), 'num': int(new[1])}
    dict['new'] = dict_new
    
    if(len(list) > 1):
        dict['in'] = list[1]
    else:
        dict['in'] = ""
        
    return dict

def diff_hunk_divider(diff_content):
    list = [] # list of dictonaries, each dictonary {old: {start: , num: },new:{start: , num: },in: ,content: }
    lines = diff_content.split('\n')
    i = 0 
    while(i < len(lines)):
        if lines[i].startswith("@@"):
            dict = diff_header_process(lines[i])
            i = i+1
            content_str = "" 
            changed_no_of_next_lines = False
            while(i < len(lines) and not(lines[i].startswith("@@"))):
                if(lines[i] != "\ No newline at end of file"): # they added this fucking line at end of some diff strings
                    changed_no_of_next_lines = True
                    content_str += lines[i]+"\n"
                i = i+1
            if(changed_no_of_next_lines):
                content_str = content_str[:len(content_str)-1] # removing last added \n
            dict["content"] = content_str
            list.append(dict)
        else:
            i = i+1
    
    return list 

python_common_tokens = ['abs','delattr','hash','memoryview','set','all','dict','help','min','setattr','any','dir','hex','next','slice','ascii','divmod','id','object','sorted','bin','enumerate','input','oct','staticmethod','bool','eval','int','open','str','breakpoint','exec','isinstance','ord','sum','bytearray','filter','issubclass','pow','super','bytes','float','iter','print','tuple','callable','format','len','property','type','chr','frozenset','list','range','vars','classmethod','getattr','locals','repr','zip','compile','globals','map','reversed','__import__','complex','hasattr','max','round','False','await','else','import','passNone','break','except','in','raise','True','class','finally','is','return','and','continue','for','lambda','try','as','def','from','nonlocal','while','assert','del','global','not','with','async','elif','if','or','yield', 'self']

def preprocess_code_line(code, remove_python_common_tokens=False):
    code = code.replace('(',' ').replace(')',' ').replace('{',' ').replace('}',' ').replace('[',' ').replace(']',' ').replace('.',' ').replace(':',' ').replace(';',' ').replace(',',' ').replace(' _ ', '_')
    code = re.sub('``.*``','<STR>',code)
    code = re.sub("'.*'",'<STR>',code)
    code = re.sub('".*"','<STR>',code)
    code = re.sub('\d+','<NUM>',code)
    code = re.sub('#.*','',code)
    
    if remove_python_common_tokens:
        new_code = ''

        for tok in code.split():
            if tok not in python_common_tokens:
                new_code = new_code + tok + ' '
            
        return new_code.strip()
    
    else:
        return code.strip()

def find_ele_nparray(np_array,ele):
    for i in range(0,len(np_array)):
        if(ele == np_array[i]):
            return i
    return -1

def data_convertor(given_repo,given_commit_hash,diff_content,buggy_lines):
    buggy_lines = np.array(buggy_lines)
    repo = []
    commit_hash = []
    code_change = []
    change_type = []
    is_buggy_line = []
    code_change_remove_common_tokens = []
    lines_nos = []
    if(len(diff_content) == 0 ): #bug some(258) diff_content are empty so 0-1 out of bounds error SCAMING WITH EMPTY DATA
        return repo, commit_hash, code_change, change_type, is_buggy_line, code_change_remove_common_tokens, lines_nos
    diff_content = diff_content[:len(diff_content)-1] # removing that last extra "\n" 
    list = diff_hunk_divider(diff_content)

    for hunk_dict in list:
        # print("for", end = ' ')
        old_start_line = hunk_dict["old"]["start"]
        old_no_lines = hunk_dict["old"]["num"]
        new_start_line = hunk_dict["new"]["start"]
        new_no_lines = hunk_dict["new"]["num"]
        change_in = hunk_dict["in"]
        changed_content = hunk_dict["content"]
        changed_lines = changed_content.split("\n")
        no_lines = 0 # consider it a index
        i = 0
        while( i < len(changed_lines)):
            buggy = False
            # print("while", end = " ")
            line = changed_lines[i]
            if line.startswith("-"):
                # print("-")
                # print("if", end = ' ')
                line = line[1:]
                idx = line.find('"""')
                if(idx >= 0):                    # found multi line comment need to find the end
                    temp = line[idx+3:]
                    rem1 = line[:idx]
                    idx = temp.find('"""')       # checking if multi line ends in the same line else find the end in next lines
                    if(idx < 0):
                        i += 1
                        while(i < len(changed_lines)):
                            idx = changed_lines[i].find('"""')
                            if(idx > 0):
                                break
                            i += 1
                        if(idx > 0):
                            rem2 = changed_lines[i][idx+3:]
                            line = rem1 + "<STR>" + rem2
                        else:
                            line = rem1 + "<STR>"
                            
                repo.append(given_repo)
                commit_hash.append(given_commit_hash)
                code_change.append(line)
                change_type.append("removed")
                if(buggy):
                    is_buggy_line.append(1.0)
                else:    
                    is_buggy_line.append(0.0)
                code_change_remove_common_tokens.append(preprocess_code_line(line,True))
                lines_nos.append(-1)

            elif line.startswith("+"):
                # print("else",end = ' ')
                lines_nos.append(new_start_line+no_lines)
                # print("+",end="")
                # print(new_start_line+no_lines)
                if(find_ele_nparray(buggy_lines,new_start_line+no_lines) >= 0):
                    buggy = True
                line = line[1:]
                idx = line.find('"""')
                if(idx >= 0):                    # found multi line comment need to find the end
                    temp = line[idx+3:]
                    rem1 = line[:idx]
                    idx = temp.find('"""')       # checking if multi line ends in the same line else find the end in next lines
                    if(idx < 0):
                        i += 1
                        no_lines += 1
                        # print(new_start_line+no_lines)
                        buggy = buggy or find_ele_nparray(buggy_lines,new_start_line+no_lines) >= 0
                        while(i < len(changed_lines)):
                            idx = changed_lines[i].find('"""')
                            if(idx > 0):
                                break
                            i += 1
                            no_lines += 1
                            # print(new_start_line+no_lines)
                            buggy = buggy or find_ele_nparray(buggy_lines,new_start_line+no_lines) >= 0
                        if(idx > 0):
                            rem2 = changed_lines[i][idx+3:]
                            line = rem1 + "<STR>" + rem2
                        else:
                            line = rem1 + "<STR>"
                else:
                    no_lines += 1
                            
                repo.append(given_repo)
                commit_hash.append(given_commit_hash)
                code_change.append(line)
                change_type.append("added")
                if(buggy):
                    is_buggy_line.append(1.0)
                else:    
                    is_buggy_line.append(0.0)
                code_change_remove_common_tokens.append(preprocess_code_line(line,True))
                
            else:
                # print("*")
                no_lines += 1
            i += 1
            

    return repo, commit_hash, code_change, change_type, is_buggy_line, code_change_remove_common_tokens, lines_nos
            
start = time.time()
repo_full = []
commit_hash_full = []
code_change_full = []
change_type_full = []
is_buggy_line_full = []
code_change_remove_common_tokens_full = []
lines_nos_full = []
for i in range(0,10000):
    repo, commit_hash, code_change, change_type, is_buggy_line, code_change_remove_common_tokens, lines_nos = data_convertor(test_df["repo"][i],test_df["commit"][i],test_df["content"][i],test_df["lines"][i])
    repo_full += repo
    commit_hash_full += commit_hash
    code_change_full += code_change
    change_type_full +=change_type 
    is_buggy_line_full += is_buggy_line
    code_change_remove_common_tokens_full += code_change_remove_common_tokens
    lines_nos_full+=lines_nos
dict = {"repo": repo_full, "commit_hash": commit_hash_full, "code_change": code_change_full,"change_type":change_type_full, "is_buggy_line":is_buggy_line_full, "lines_nos": lines_nos_full, "code_change_remove_common_tokens":code_change_remove_common_tokens_full}
end = time.time()
print(end -start)
line_level_jit_test = pd.DataFrame(dict)

def pre_process_diff(diff_content):    
    lines = diff_content.split("\n")
    removed_lines = []
    added_lines = []
    for line in lines:
        if line.startswith("@@"):
            continue  # Skip header lines
    
        if line.startswith("+"):
            line = line[line.index('+')+1:]
            if line.startswith("#"): # Ignoring the comment line as it doesn't contribute to a bug
                continue
            hashtag_idx = line.find("#") # removing the comment line at the end of the line 
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            added_lines.append(preprocess_code_line(line,remove_python_common_tokens=True))
        elif line.startswith("-"):
            line = line[line.index('-')+1:]
            if line.startswith("#"): # Ignoring the comment line as it doesn't contribute to a bug
                continue
            hashtag_idx = line.find("#") # removing the comment line at the end of the line 
            if hashtag_idx > 0:
                line = line[:hashtag_idx]
            removed_lines.append(preprocess_code_line(line,remove_python_common_tokens=True))

    added_code = ' \n '.join(list(set(added_lines)))
    removed_code = ' \n '.join(list(set(removed_lines)))
    return added_code + " " + removed_code

line_level_jit_test.to_csv('./line_level_jit_test.csv')

