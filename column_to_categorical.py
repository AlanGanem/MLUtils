def column_to_categorical(df ,columns,column_names =None, sep = '_'):
    concat_df = 0
    split_columns = [i.split(sep) for i in columns]
    levels_dict={index:[] for index,_ in enumerate(split_columns)}
    for col_name in split_columns:
        for index,value in enumerate(col_name):
            df[index] = 0
            if value not in levels_dict[index]:
                levels_dict[index].append(value)
                
    levels_dict = {key:value for key,value in levels_dict.items() if value != []}
    combinations = list(itertools.product(*list(levels_dict.values())))
    i=0
    for comb in combinations:
        combination = sep.join(comb)
        if combination in columns:    
            for level, value in enumerate(comb): 
                df[level] = value
            df['Value'] = df[combination]
            
            if i == 0:
                concat_df = df.drop(columns = columns).copy()
                i+=1
            else:
                concat_df = pd.concat([concat_df,df.drop(columns = columns)],axis = 0)
    
    if column_names:
        assert len(column_names) == len(levels_dict.keys())
        names_dict = {i:name for i,name in enumerate(column_names)}
        concat_df = concat_df.rename(columns = names_dict)
    
    index_name = concat_df.index.name
    
    return concat_df
