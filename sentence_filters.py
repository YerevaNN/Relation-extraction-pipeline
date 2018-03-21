def multiword(t, value):
    if value == 0:
        return True
    filtered = [e for e in t['entities'] if e.strip().find(' ') > 0 or e.strip().find('-') > 0]
    if value == +1:
        return len(filtered) > 0
    if value == -1:
        return len(filtered) == 0
    
    
def tags(t, values):
    if values == 'false':
        return 'tags' not in t
    
    if values == 'true':
        return 'tags' in t
    
    if 'tags' not in t:
        return False
    
    
    # print('!!', t['tags'], values)
    
    values = values.split(',')
    q = 0
    for value in values:
        tag = value[:-2]
        label = value[-2:]
        if label == '+1':
            # positive filter
            if tag in t['tags']:
                q += 1
        elif label == '-1':
            if tag not in t['tags']:
                q += 1
        else:
            raise Exception("AAA")
    
    return q == len(values)