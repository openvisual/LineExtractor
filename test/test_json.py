# -*- coding: utf-8 -*-

import json
#data = {'people':[{'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'}]}
data = {'name': 'Scott', 'website': 'stackabuse.com', 'from': 'Nebraska'}
aaa = json.dumps(data, indent=4)
print( aaa )

with open( 'c:/temp/data.json', 'w') as f:
    json.dump(data, f, indent=4)
pass
