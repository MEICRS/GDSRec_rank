# 
A PyTorch implementation of GDSRec for Ranking Task

1. Preprocess dataset. Two pkl files named dataset and list should be generated in the respective folders of the dataset.
```bash
python preprocess.py
```

2. Run main.py file to train the model. You can configure some training parameters through the command line. 
```bash
python main.py --test=False
```

3. Run main.py file to test the model.
```bash
python main.py --test=True
```

The hyper-parameter "cut_off" in ratingt_data.py corresponds to the parameter "F" in the papaer. It controls the threshold of rating for splitting the data.
