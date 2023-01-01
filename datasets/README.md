## Prepare Datasets for Mask2Former

You should get the LIP, PASCAL-Person-Part, CIHP, MHP-v2 from original website, and put them in `M2FP/datasets/` folder.


### Expected dataset structure for LIP:

```
lip/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
  Validation/
    Images/
    Category_ids/
```


### Expected dataset structure for PASCAL-Person-Part:
```
pascal-person-part/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
    Instance_ids/       # all training part instance labels
    Human_ids/          # all training human instance labels
  Testing/
    Images/
    Category_ids/
    Instance_ids/
    Human_ids/
```


### Expected dataset structure for CIHP:
```
cihp/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
    Instance_ids/       # all training part instance labels
    Human_ids/          # all training human instance labels
  Validation/
    Images/
    Category_ids/
    Instance_ids/
    Human_ids/
```

### Expected dataset structure for MHP-v2:
```
mhpv2/
  Training/
    Images/             # all training images
    Category_ids/       # all training category labels
    Instance_ids/       # all training part instance labels
    Human_ids/          # all training human instance labels
  Validation/
    Images/
    Category_ids/
    Instance_ids/
    Human_ids/
```