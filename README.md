# 
A PyTorch implementation of GDSRec for Ranking Task

1. Preprocess dataset. Two pkl files named dataset and list should be generated in the respective folders of the dataset.
```bash
python preprocess.py
```

3. Run main.py file to train the model. You can configure some training parameters through the command line. 
```bash
python main.py --test_flag=False
```

4. Run main.py file to test the model.
```bash
python main.py --test_flag=True
```
