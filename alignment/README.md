# Monitor the alignement of the MVTX detector

### Requirements:

1. copy these two files `Makefile` and `RotTransFit.cpp` to your path.
details about the fit routine found here: https://github.com/ymei/BodyRotTransFit

2. Install some useful python libraries:

```
pip install pandas numpy seaborn matplotlib plotly circle_fit
```

for help run:
```
python3 mvtxPositionTest.py --h
```

### Example:

```
python3 mvtxPositionTest.py --input_model model/MITSEW_L2.txt --input_cmm cmm/SouthEndWheel1_L2_TEST_011922.dat --r 1.0 --a 0.1 --b 0.0 --g 0.1 --xt 0.1 --yt 0.1 --zt 0.1 --xt0 353.5 --fidList 3 4 --pointList 1 2 --sigmaList 0.1 0.1 100 100
```