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
    
    graphs = []
    graphs.append(dict(data=genres_graph, layout=genres_layout))
    
    return graphs