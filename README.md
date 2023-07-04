# MT_GCNN

The source code of the manuscript entitle "Move and remove: Multi-task learning for building simplification in vector maps with a graph convolutional neural network"

# Requirements

* python 3.7
* pytorch 1.11.0
* torch-geometric 2.0.3

# Data description

The building graph features and labels are stored in the directory *data/input/* with numpy *.npy* file. The description for each columan of the file is as follows:

* Column 0: osm id of the building corresponding to the vertice
* Column 1: vertex id of the vertice
* Column 2: longitude of the vertice
* Column 3: latitude of the vertice
* Column 4: normalized longitude of the vertice
* Column 5: normalized latitude of the vertice
* Column 6: turning angle of the vertice
* Column 7: convexity of the vertice
* Column 8: preceeding edge length of the vertice 
* Column 9: succeeding edge length of the vertice
* Column 10: **the removal label**
* Column 11: **the movement label** along the preceeding edge
* Column 12: **the movement** label along the succeeding edge

# Usage

Config the input directory and hyperparameters in main.py and run it. In case of understanding the proposed MT_GCNN model, check it in models.py with the class *BuildingGenModel*.

* **data/input**: the directory where vertex feature and edge adjacency files are input to the model
* **data/output**: the directory where the trained model (Bldgs_Gen_64_1.pkl) and the predicted vertex labels (the point file: Bldgs_Gen_prediction.shp) are stored
* **Noete**: to make the final simplified buildings, please use the *reconstruct_polygons* function in *utils.py* to reconstruct polygons based on the output point file *Bldgs_Gen_prediction.shp*

# Citation

```
@article{zhou2023move,
  title={Move and remove: Multi-task learning for building simplification in vector maps with a graph convolutional neural network},
  author={Zhou, Zhiyong and Fu, Cheng and Weibel, Robert},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={202},
  pages={205--218},
  year={2023},
  publisher={Elsevier}
}
```
