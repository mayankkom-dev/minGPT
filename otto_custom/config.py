class CFG:
    num_workers=2
    path="../input/fb3-conv1dpooler/"
    config_path=path+'config.pth'
    model="sshleifer/distilbart-xsum-9-6" 
    gradient_checkpointing=False
    batch_size=32
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed=69
    n_fold=4
    trn_fold=[0, 1, 2, 3]