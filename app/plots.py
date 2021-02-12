import plotly.graph_objects as go

def return_graphs(df):
    ''' Module to create Plotly figures '''
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    genres_graph = []
    
    genres_graph.append(
        go.Bar(
            x=genre_names,
            y=genre_counts
        )
    )
    
    genres_layout = dict(
        {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    )
    
    
    # extract count for each category occurance
    categ_counts = df[df.columns[4:]].sum().sort_values(ascending=True)
    categ_names = list(categ_counts.index)
    
    # create category visual
    categ_graph = []
    
    categ_graph.append(
        go.Bar(
            x=categ_counts,
            y=categ_names,
            orientation='h'
        )
    )
    
    categ_layout = dict(
        {
            'title': 'Category Occurance Count',
            
            'yaxis': {
                'title': 'Category'
            },
            
            'xaxis': {
                'title': 'Count'
            },
            
            "margin": {
                "pad": 12,
                "l": 160,
                "r": 40,
                "t": 80,
                "b": 40,
            },
            
            "yaxis": {"dtick": 1}
        }
    )
    
    
    graphs = []
    graphs.append(dict(data=genres_graph, layout=genres_layout))
    graphs.append(dict(data=categ_graph, layout=categ_layout))
    
    return graphs