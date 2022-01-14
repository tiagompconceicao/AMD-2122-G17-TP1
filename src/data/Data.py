import Orange as DM 

class Data:
    def __init__(self):
        pass

    #Create a dataset from file
    def load(self, fileName):
        try:
            dataset = DM.data.Table( fileName )
        except:
            exit()
        return dataset

    def convertArraytoTable(self, domain, table):
        return DM.data.Instance(domain, table)

    #Create a dataset from a array
    def createTable(self,table,classes,domain):
        return DM.data.Table.from_numpy(domain=domain,X=table,Y=classes)

    #Create a dataset from another dataset
    def convertTable(self, table, domain):
        return DM.data.Table.from_table(domain,table)

    #Convert dataset into a array
    def convertTableToArray(self, node,domain):
        table = []
        for data in node:
            attributeValues = []
            for i in range(len(domain.variables) - 1):
                attributeValues.append(data[i])
            table.append(attributeValues)
        return table

    #Create domain
    def createDomain(self,x,y):
        try:
            domain = DM.data.Domain(x,y)
        except:
            exit()
        return domain

    #Get Variable from dataset using a given name
    def get_variableFrom_str(self, dataset, str_name ):
        variable_list = dataset.domain.variables
        for variable in variable_list:
            if( variable.name == str_name ): return variable
        return None
    