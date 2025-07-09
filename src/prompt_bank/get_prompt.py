
'''
samples = [
    {"Probability Matrix": [], "MAE": , "MAPE": , "RMSE": },
    ...
]

'''

def get_prompt(dataset_name, samples, current_epoch, total_epoch, rate=0.8):
    exploring_rounds = total_epoch * rate
    if "PEMS" in dataset_name:
        sys_prompt = f"""
    Please select appropriate modules for the following deep learning task. 
    Task description: 
        This is a traffic forecasting dataset, consisting of hundreds of sensors monitoring the traffic indices around the city. 
        The goal is to predict future traffic indices according to history indices for each sensor. 
        The deep learning task should properly capture spatial and temporal information.
        We have three kinds of modules with spatial-then-temporal, temporal-then-spatial and spatial-temporal-parallely.
        Three metrics to evaluate the results predicted by the model, including MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and RMSE (Root Mean Square Error)
    Here are some examples of this task and their corresponding module selections:
        {samples}
    Your task: 
        First, analyze the characteristics and requirements of this task.  
        Then, consider the applicability of different modules in this task. For example, the spatial-then-temporal prioritize spatial information, while temporal-then-spatial prioritize temporal information. Spatial-temporal-parallely module deal with spatial and temporal information parallely.
        Next, observe the examples, find out how to select appropriate modules to make MSE, MAE and RMSE lower.
        Considering these factors comprehensively, for the six layers, select the most suitable combination of modules for each layer and explain your choice.
    Provide no additional text in response, Format output in JSON as {{ "Combination of modules": "[choice for layer1, choice for layer2, choice for layer3, choice for layer4, choice for layer5, choice for layer6]", "Explanation": "explain your choice"}}
    """
    else:
        if current_epoch < exploring_rounds:
            sys_prompt = f"""
    Please select appropriate modules for the following deep learning task. 
    Background description: 
        The dataset: This is a traffic forecasting dataset, consisting of hundreds of sensors monitoring the traffic indices around the city. The goal is to predict future traffic indices according to history indices for each sensor. 
        The options: The deep learning task should properly capture spatial and temporal information with a certain combination of several layers, and each layer can choose one of the three kinds of modules, namely spatial-then-temporal, temporal-then-spatial and spatial-temporal-parallely. 
        The architecture: The whole deep learning architecture is a directed acyclic graph with four nodes and six layers, and the preceding nodes are connected to each subsequent nodes with a layer. e.g. Node 1 connect Nodes 2 with Layer 1, Node 1 connect Nodes 3 with Layer 2, ...
        The metrics: Three metrics to evaluate the results predicted by the model, including MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and RMSE (Root Mean Square Error)
    Here are some historical samples obtained in previous rounds to guide you to select more suitable combination of modules (sorted by MAE):
        {samples}
    Your task: 
        First, analyze the background description of this task.  
        Then, observe the samples from previous rounds, consider the applicability of different modules in this task. 
        Next, considering these factors comprehensively, for the six layers, try to design a new combination that is not exsited in historical samples to potentially achieve better behaviours and explain your choice. You have {total_epoch} rounds to try, and this is the {current_epoch + 1} round.  
        Provide no additional text in response, Format output in JSON as {{ "Combination of modules": {{"Layer_1": "choice for layer_1", "Layer_2": "choice for layer_2", "Layer_3": "choice for layer_3", "Layer_4": "choice for layer_4", "Layer_5": "choice for layer_5", "Layer_6": "choice for layer_6"}}, "Explanation": "explain your choice"}}
    """
        else:
            sys_prompt = f"""
    Please select appropriate modules for the following deep learning task. 
    Background description: 
        The dataset: This is a traffic forecasting dataset, consisting of hundreds of sensors monitoring the traffic indices around the city. The goal is to predict future traffic indices according to history indices for each sensor. 
        The options: The deep learning task should properly capture spatial and temporal information with a certain combination of several layers, and each layer can choose one of the three kinds of modules, namely spatial-then-temporal, temporal-then-spatial and spatial-temporal-parallely. 
        The architecture: The whole deep learning architecture is a directed acyclic graph with four nodes and six layers, and the preceding nodes are connected to each subsequent nodes with a layer. e.g. Node 1 connect Nodes 2 with Layer 1, Node 1 connect Nodes 3 with Layer 2, ...
        The metrics: Three metrics to evaluate the results predicted by the model, including MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and RMSE (Root Mean Square Error)
    Here are some historical samples obtained in previous rounds to guide you to select more suitable combination of modules (sorted by MAE):
        {samples}
    Your task: 
        First, analyze the background description of this task.  
        Then, observe the samples from previous rounds, consider the applicability of different modules in this task. 
        Next, considering these factors comprehensively, for the six layers, try to find the best combination according to previous tried samples to make MSE, MAE and RMSE lower and explain your choice. You have {total_epoch} rounds to try, and this is the {current_epoch + 1} round. 
    Provide no additional text in response, Format output in JSON as {{ "Combination of modules": {{"Layer_1": "choice for layer_1", "Layer_2": "choice for layer_2", "Layer_3": "choice for layer_3", "Layer_4": "choice for layer_4", "Layer_5": "choice for layer_5", "Layer_6": "choice for layer_6"}}, "Explanation": "explain your choice"}}
        """
    return sys_prompt

def get_new_state_prompt(samples, current_layers, current_layer_num):
    sys_prompt = f'''
    Please select appropriate modules for the following deep learning task. 
    Background description: 
        The dataset: This is a traffic forecasting dataset, consisting of hundreds of sensors monitoring the traffic indices around the city. The goal is to predict future traffic indices according to history indices for each sensor. 
        The options: The deep learning task should properly capture spatial and temporal information with a certain combination of several layers, and each layer can choose one of the three kinds of modules, namely spatial-then-temporal, temporal-then-spatial and spatial-temporal-parallely. 
        The architecture: The whole deep learning architecture is a directed acyclic graph with four nodes and six layers, and the preceding nodes are connected to each subsequent nodes with a layer. e.g. Node 1 connect Nodes 2 with Layer 1, Node 1 connect Nodes 3 with Layer 2, ...
        The metrics: Three metrics to evaluate the results predicted by the model, including MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and RMSE (Root Mean Square Error)
    Here are some historical samples obtained in previous rounds to guide you to select more suitable combination of modules (sorted by MAE):
        {samples}
    You have chosen {current_layer_num} layers, they are: {current_layers}.
    Your task: 
        First, analyze the background description of this task.  
        Then, observe the historical samples and the layers you have chosen. 
        Next, considering these factors comprehensively, try to choose the next one layer based on your current chosen layers, that should not be too similar to the history samples to potentially achieve better behaviours, and explain the reason. 
    Provide no additional text in response, Format output in JSON as {{"New layer": "Your choice for new layer", "Explanation": "explain your choice"}}
    '''
    return sys_prompt
def get_evaluate_prompt(samples, current_layers, current_layer_num):
    sys_prompt = f'''
    Please select appropriate modules for the following deep learning task. 
    Background description: 
        The dataset: This is a traffic forecasting dataset, consisting of hundreds of sensors monitoring the traffic indices around the city. The goal is to predict future traffic indices according to history indices for each sensor. 
        The options: The deep learning task should properly capture spatial and temporal information with a certain combination of several layers, and each layer can choose one of the three kinds of modules, namely spatial-then-temporal, temporal-then-spatial and spatial-temporal-parallely. 
        The architecture: The whole deep learning architecture is a directed acyclic graph with four nodes and six layers, and the preceding nodes are connected to each subsequent nodes with a layer. e.g. Node 1 connect Nodes 2 with Layer 1, Node 1 connect Nodes 3 with Layer 2, ...
        The metrics: Three metrics to evaluate the results predicted by the model, including MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and RMSE (Root Mean Square Error)
    Here are some historical samples obtained in previous rounds to guide you to select more suitable combination of modules (sorted by MAE):
        {samples}
    You have chosen {current_layer_num} layers, they are: {current_layers}.
    Your task: 
        First, analyze the background description of this task.  
        Then, observe the historical samples and the layers you have chosen. 
        Next, judge if it is possible that the layers you have chosen will lead into a better result and explain the reason.
    Provide no additional text in response, Format output in JSON as {{"Judgement": "possible or impossible", "Explanation": "explain your judgement"}}
    '''
    return sys_prompt

'''
First, analyze the characteristics and requirements of this task.  
        Then, consider the applicability of different modules in this task. For example, the spatial-then-temporal prioritize spatial information, while temporal-then-spatial prioritize temporal information. Spatial-temporal-parallely module deal with spatial and temporal information parallely.
        Next, observe the samples, find out how to select appropriate modules to make MSE, MAE and RMSE lower.
'''