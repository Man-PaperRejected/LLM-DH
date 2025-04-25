class PR():
    def __init__(self):
        
        self.data = [1,2,3,4]

pr = PR()
data = pr.data

data[2]=999
print(pr.data)