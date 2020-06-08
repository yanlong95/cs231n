# Project Title

In this paper, a new approach that classifies the building instance category by using the street view images is presented. Since every building is unique, the classical image classification architectures have limited performance on this particular task. Therefore, the paper proposes a new model that takes account of both building’s design and region. In order to achieve this, a multi-task training technique considering the building's region label is implemented on top of a classical ResNet50 model. 



## Running the Code
put the dataset in the 'dataset' directory 

### Train and evaluate the model



```
python main_v2.py
```

### Test the model

```
python evaluate_v2.py
```

### build the city map

```
python city_map_builder.py
```


## Deployment

Add additional notes about how to deploy this on a live system



## Authors

* **Zonglin Li** - [Zonglin](https://github.com/zjackli)
* **Yanlong Ma** - [Yanlong](https://github.com/yanlong95)

