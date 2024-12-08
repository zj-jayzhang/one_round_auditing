This is an unofficial implementation of the paper [Privacy Auditing with One (1) Training Run](https://arxiv.org/abs/2305.08846).

# Requirements

Install and activate the conda environment in environment.yml:

```bash
micromamba env create -f environment.yml -n privacy-audit
micromamba activate privacy-audit
```

# How to run

1. first train the model using the following command:
```bash
bash scripts/dp_train.sh
```

Note: we find that the training process heavily depends on the random seed, so the result may vary. Thus, we train 32 models and report the average epsilon.

At default, the epsilon is set to 100, which is a very large value. You can change it to a smaller value to 7 in the `scripts/dp_train.sh` file.

This code only supports the Input Space Attacks, and the Gradient Space Attacks are not implemented. 



2. then run the following command to audit the model:
```bash
bash scripts/dp_audit.sh
```



# Results
Here at default, the epsilon is set to 100. 
$k_+$ is set to 80, and $k_-$ is set as range(10, 200, 10). You may tune the hyperparameters to get better results. 

The result is as follows:

```markdown

shadow model 0, best eps: 0.043622709810733795
shadow model 1, best eps: 0.2546472381800413
shadow model 2, best eps: 0.08018947206437588
shadow model 3, best eps: 0.08836076781153679
shadow model 4, best eps: 0
shadow model 5, best eps: 0
shadow model 6, best eps: 0.11595284007489681
shadow model 7, best eps: 0.38184524327516556
shadow model 8, best eps: 0.3511759154498577
shadow model 9, best eps: 0
shadow model 10, best eps: 0.06948579754680395
shadow model 11, best eps: 0.13328915182501078
shadow model 12, best eps: 0.06760371197015047
shadow model 13, best eps: 0.3178618336096406
shadow model 14, best eps: 0.21079052612185478
shadow model 15, best eps: 0.01702991221100092
shadow model 16, best eps: 0.009206552058458328
shadow model 17, best eps: 0
shadow model 18, best eps: 0.16490325704216957
shadow model 19, best eps: 0
shadow model 20, best eps: 0.15118373278528452
shadow model 21, best eps: 0.1578795202076435
shadow model 22, best eps: 0
shadow model 23, best eps: 0.10458909720182419
shadow model 24, best eps: 0.16984295472502708
shadow model 25, best eps: 0.1995136495679617
shadow model 26, best eps: 0.3509875973686576
shadow model 27, best eps: 0
shadow model 28, best eps: 0.3325097216293216
shadow model 29, best eps: 0.3528527766466141
shadow model 30, best eps: 0.2820653775706887
average eps: 0.1421738502178942

```

Since the epsilon is set to 100, the result is pretty loose here.


# Contact

If you find anything wrong or have any questions, please feel free to contact me at `jie.zhang@inf.ethz.ch'.  I’d be happy to connect!

The code is based on the our CCS 2024 paper [Evaluations of Machine Learning Privacy Defenses are Misleading](https://arxiv.org/abs/2404.17399). If you find anything unclear, you may refer to the paper and code for more details.

